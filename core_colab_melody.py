import torch
import torchaudio
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from transformers import pipeline
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import logging
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import silhouette_score

# Silenzia log di sistema
logging.getLogger("audiocraft").setLevel(logging.ERROR)

# ==========================================
# 1. WRAPPER MODELLO (Melody Edition)
# ==========================================
class MusicGenWrapper:
    def __init__(self, size='facebook/musicgen-melody', duration=5):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            self.device = "cpu"
        
        print(f"🚀 MusicGen Melody ({size}) | Device: {self.device} | Durata: {duration}s")
        self.model = MusicGen.get_pretrained(size)
        self.model.set_generation_params(duration=duration)

    def load_melody(self, path):
        """Carica, converte a mono e ricampiona la melodia per MusicGen."""
        if not os.path.exists(path):
            print(f"⚠️ Melodia non trovata: {path}")
            return None, None
            
        melody, sr = torchaudio.load(path)
        
        # Converti a Mono se necessario (MusicGen lo gestisce, ma meglio essere espliciti)
        if melody.dim() == 2 and melody.shape[0] > 1:
            melody = melody.mean(dim=0, keepdim=True)
            
        # Il modello gestisce il resampling internamente in generate_with_chroma,
        # ma passiamo il tensore grezzo e il sample rate originale.
        return melody.to(self.device), sr

    def generate(self, prompt, filename=None, melody_path=None):
        """
        Genera audio. Se melody_path è fornito, usa generate_with_chroma.
        """
        if melody_path:
            melody_wav, melody_sr = self.load_melody(melody_path)
            if melody_wav is not None:
                # Aggiungi dimensione batch [1, C, T]
                if melody_wav.dim() == 2: melody_wav = melody_wav.unsqueeze(0)
                
                # Generazione guidata da Melodia
                wav = self.model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=melody_wav,
                    melody_sample_rate=melody_sr,
                    progress=False
                )
            else:
                # Fallback se melodia non caricata
                print(f"⚠️ Fallback a testo-solo per '{prompt}'")
                wav = self.model.generate([prompt], progress=False)
        else:
            # Generazione solo testo
            wav = self.model.generate([prompt], progress=False)

        if filename:
            if filename.endswith(".wav"): filename = filename[:-4]
            folder = os.path.dirname(filename)
            if folder: os.makedirs(folder, exist_ok=True)
            audio_write(filename, wav[0].cpu(), self.model.sample_rate, strategy="loudness", loudness_headroom_db=16)
        
        return wav

# ==========================================
# 2. ACTIVATION HOOK (Identico)
# ==========================================
class ActivationHook:
    def __init__(self, module):
        self.module = module
        self.handle = None
        self.activations = [] 

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple): output = output[0]
        # FIX CFG: Prendi solo il primo elemento (condizionato)
        if output.shape[0] > 1: output = output[0:1]
        self.activations.append(output.detach().cpu())

    def register(self):
        self.handle = self.module.register_forward_hook(self.hook_fn)
        self.activations = [] 

    def remove(self):
        if self.handle: self.handle.remove()
    
    def get_mean_vector(self):
        if not self.activations: return None
        full = torch.cat(self.activations, dim=1)
        return full.mean(dim=1).squeeze()

# ==========================================
# 3. DYNAMIC STEERING
# ==========================================
class DynamicSteering:
    def __init__(self, module, steering_vector):
        self.module = module
        self.handle = None
        self.alpha = 0.0
        try: device = next(module.parameters()).device
        except: device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.vector = steering_vector.to(device).float()
        if self.vector.dim() == 1: self.vector = self.vector.view(1, 1, -1)
        elif self.vector.dim() == 2: self.vector = self.vector.unsqueeze(1)

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple): h, other = output[0], output[1:]
        else: h, other = output, ()
        h_new = (h + (self.alpha * self.vector)) / (1 + abs(self.alpha))
        if other: return (h_new,) + other
        return h_new

    def apply(self, coefficient=1.0):
        self.alpha = coefficient
        if self.handle is None:
            self.handle = self.module.register_forward_hook(self.hook_fn)
        
    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

# ==========================================
# 4. DATASET EXTRACTOR (Con Supporto Melodia)
# ==========================================
class DatasetExtractor:
    def __init__(self, model_wrapper, layers=[14]):
        self.mg = model_wrapper
        if isinstance(layers, int): layers = [layers]
        self.target_layers_indices = layers
        
        self.hooks = {}
        for idx in self.target_layers_indices:
            # Nota: MusicGen Melody potrebbe avere più layer (48).
            # Assicurati che l'indice esista.
            try:
                layer_module = self.mg.model.lm.transformer.layers[idx]
                self.hooks[idx] = ActivationHook(layer_module)
            except IndexError:
                print(f"❌ ERRORE: Layer {idx} non esiste (Max {len(self.mg.model.lm.transformer.layers)-1})")
        
    def extract(self, csv_path, save_path, audio_output_dir=None, sep=';', use_pca=False):
        print(f"🏭 Multi-Extraction Melody {self.target_layers_indices} -> {save_path}")
        
        try: df = pd.read_csv(csv_path, sep=sep)
        except: print("❌ Error reading CSV"); return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if audio_output_dir: os.makedirs(audio_output_dir, exist_ok=True)

        layer_sums = {idx: None for idx in self.target_layers_indices}
        count = 0

        for h in self.hooks.values(): h.register()

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                p_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                p_neg = str(row.get('negative_prompt', row.iloc[1])).strip()
                
                # LEGGI MELODIA (Se presente nel CSV)
                melody_file = None
                if 'melody_path' in row and pd.notna(row['melody_path']):
                    melody_file = str(row['melody_path']).strip()
                
                pid = str(row.get('ID', index)).strip()

                # --- POSITIVO ---
                f_pos = os.path.join(audio_output_dir, f"{pid}_pos") if audio_output_dir else None
                self.mg.generate(p_pos, filename=f_pos, melody_path=melody_file)
                vecs_pos = {idx: h.get_mean_vector() for idx, h in self.hooks.items()}
                for h in self.hooks.values(): h.activations = [] 

                # --- NEGATIVO ---
                f_neg = os.path.join(audio_output_dir, f"{pid}_neg") if audio_output_dir else None
                self.mg.generate(p_neg, filename=f_neg, melody_path=melody_file)
                vecs_neg = {idx: h.get_mean_vector() for idx, h in self.hooks.items()}
                for h in self.hooks.values(): h.activations = [] 

                # --- CALCOLO ---
                valid_pair = True
                current_diffs = {}
                for idx in self.target_layers_indices:
                    v_p = vecs_pos[idx]
                    v_n = vecs_neg[idx]
                    if v_p is None or v_n is None: 
                        valid_pair = False; break
                    
                    diff = v_p - v_n
                    diff = diff / (diff.norm() + 1e-8) # Normalizzazione Locale
                    current_diffs[idx] = diff

                if valid_pair:
                    for idx, diff in current_diffs.items():
                        if layer_sums[idx] is None: layer_sums[idx] = diff
                        else: layer_sums[idx] += diff
                    count += 1

            except Exception as e: print(f"Err {index}: {e}")

        for h in self.hooks.values(): h.remove()

        # Media Finale
        final_vectors_dict = {}
        if count > 0:
            print(f"🧮 Media su {count} coppie...")
            for idx, total_sum in layer_sums.items():
                if total_sum is None: continue
                mean_vec = total_sum / count
                mean_vec = mean_vec / (mean_vec.norm() + 1e-8)
                if mean_vec.dim() == 1: mean_vec = mean_vec.unsqueeze(0)
                final_vectors_dict[idx] = mean_vec

            torch.save(final_vectors_dict, save_path)
            print(f"✅ Saved Multi-Layer Dictionary: {save_path}")
        else:
            print("❌ Nessun vettore estratto.")

# ==========================================
# 5. DATASET INFERENCE (Con Supporto Melodia)
# ==========================================
class DatasetInference:
    def __init__(self, model_wrapper, layers=None):
        self.mg = model_wrapper
        # Salva i layer di default passati all'inizializzazione
        self.default_layers = layers if isinstance(layers, list) else [layers] if layers else [14]

    def run(self, prompts_file, vector_path, output_dir, alpha=1.5, active_layers=None, max_samples=None):
        """
        Args:
            alpha: float o dict (es. {'low': 0.8, 'high': 2.5})
            active_layers: list (es. [14, 15]). Se None, usa self.default_layers.
        """
        print(f"🚀 Multi-Layer Inference...")
        print(f"   Alpha Config: {alpha}")
        
        # LOGICA DI SELEZIONE LAYER:
        # Priorità: 1. active_layers (da run) -> 2. default_layers (da init) -> 3. None (usa tutto)
        target_layers = active_layers if active_layers is not None else self.default_layers
        print(f"   Target Layers Filter: {target_layers if target_layers else 'ALL LAYERS IN FILE'}")
        
        steerers = []
        try:
            data = torch.load(vector_path)
            
            if isinstance(data, dict):
                sorted_idxs = sorted(data.keys())
                
                for idx in sorted_idxs:
                    # --- FILTRO CRUCIALE ---
                    # Se abbiamo una lista target, saltiamo i layer che non ci sono
                    if target_layers is not None and idx not in target_layers:
                        continue 
                    
                    vec = data[idx]
                    target = self.mg.model.lm.transformer.layers[idx]
                    vec = vec / (vec.norm() + 1e-8)
                    if vec.dim() == 2 and vec.shape[0] > 1: vec = vec.mean(dim=0, keepdim=True)
                    
                    # --- CALCOLO ALPHA IBRIDO ---
                    current_alpha = 1.5 # Default fallback
                    if isinstance(alpha, (int, float)):
                        current_alpha = float(alpha)
                    elif isinstance(alpha, dict):
                        if idx <= 18: current_alpha = alpha.get('low', 1.0)
                        elif idx >= 19: current_alpha = alpha.get('high', 2.0)
                        else: current_alpha = 1.0
                    
                    s = DynamicSteering(target, vec)
                    s.target_alpha = current_alpha 
                    steerers.append(s)
                    print(f"   ✅ Loaded Layer {idx} (Alpha: {current_alpha})")
            
            else:
                # Fallback Single Layer
                print(f"📦 Single-Layer Vector detected.")
                vec = data
                vec = vec / (vec.norm() + 1e-8)
                current_alpha = float(alpha) if isinstance(alpha, (int, float)) else 1.5
                
                # Se single vector, lo applichiamo a tutti i target layers
                targets = target_layers if target_layers else [14]
                for idx in targets:
                    target = self.mg.model.lm.transformer.layers[idx]
                    s = DynamicSteering(target, vec)
                    s.target_alpha = current_alpha
                    steerers.append(s)
                    
        except Exception as e: print(f"❌ Vector error: {e}"); return

        if not steerers: print("❌ No steerers loaded. Check 'active_layers' list."); return

        try:
            df = pd.read_csv(prompts_file, sep=';')
            if max_samples: df = df.head(max_samples)
        except: print("❌ CSV error"); return

        os.makedirs(output_dir, exist_ok=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                pid = str(row.get('ID', row.get('id', i))).strip()
                if 'test_prompt' in row: prompt = str(row['test_prompt']).strip()
                elif 'prompt' in row: prompt = str(row['prompt']).strip()
                else: prompt = str(row.iloc[-1]).strip()
                
                melody_file = None
                if 'melody_path' in row and pd.notna(row['melody_path']):
                    melody_file = str(row['melody_path']).strip()

                base = os.path.join(output_dir, f"{pid}")

                # A. Originale
                self.mg.generate(prompt, f"{base}_orig", melody_path=melody_file)
                
                # B. Happy (Usa s.target_alpha)
                for s in steerers: s.apply(s.target_alpha)
                self.mg.generate(prompt, f"{base}_pos", melody_path=melody_file)
                for s in steerers: s.remove()
                
                # C. Sad (Usa -s.target_alpha)
                for s in steerers: s.apply(-s.target_alpha)
                self.mg.generate(prompt, f"{base}_neg", melody_path=melody_file)
                for s in steerers: s.remove()
                
            except Exception as e:
                print(f"Error {pid}: {e}")
                for s in steerers: s.remove()
                

# ==========================================
# 6. EVALUATION (Identica)
# ==========================================
class Evaluation:
    def __init__(self, audio_folder, output_dir, csv_filename, train_mode=False, label_pos="happy", label_neg="sad"):
        self.audio_folder = audio_folder
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        self.train_mode = train_mode
        self.label_pos = label_pos
        self.label_neg = label_neg
        self.candidate_labels = [label_pos, label_neg]
        self.ids = []
        self.file_map = {} 

        print(f"📊 Init Evaluation on {audio_folder}...")
        if not os.path.exists(audio_folder):
            print("❌ Folder not found!"); return

        for f in os.listdir(audio_folder):
            if f.endswith("_pos.wav"):
                base_name = f[:-8] 
                try:
                    pid = int(base_name.split('_')[0])
                    self.ids.append(pid)
                    self.file_map[pid] = base_name
                except: pass
        self.ids.sort()
        print(f"✅ Found {len(self.ids)} samples.")
        
        try: self.classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
        except: print("⚠️ Warning: CLAP load failed.")

    def _get_score(self, path):
        if not os.path.exists(path): return 0.0
        try:
            res = self.classifier(path, candidate_labels=self.candidate_labels)
            s_pos = next(x['score'] for x in res if x['label'] == self.label_pos)
            s_neg = next(x['score'] for x in res if x['label'] == self.label_neg)
            return s_pos - s_neg
        except: return 0.0

    def _create_plot(self, values, labels, title, filename_suffix, y_limit=(-1.1, 1.1)):
        plt.figure(figsize=(12, 6))
        cmap = plt.get_cmap('bwr')
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        bars = plt.bar(labels, values, color=cmap(norm(values)), edgecolor='black')
        plt.axhline(0, color='black'); plt.ylim(y_limit[0], y_limit[1])
        plt.title(title); plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if self.output_dir:
            fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(os.path.join(self.output_dir, f"plot_{fname}.png"))
        plt.show(); plt.close()

    def run_eval(self, num=20):
        target_ids = self.ids[:num]
        scores_orig, deltas_pos, deltas_neg, labels, data_csv = [], [], [], [], []

        print("--- Calcolo Score ---")
        for pid in tqdm(target_ids):
            base = self.file_map.get(pid)
            p_orig = os.path.join(self.audio_folder, f"{base}_orig.wav")
            p_pos = os.path.join(self.audio_folder, f"{base}_pos.wav")
            p_neg = os.path.join(self.audio_folder, f"{base}_neg.wav")
            
            s_orig = self._get_score(p_orig)
            s_pos = self._get_score(p_pos)
            s_neg = self._get_score(p_neg)
            
            scores_orig.append(s_orig)
            deltas_pos.append(s_pos - s_orig)
            deltas_neg.append(s_neg - s_orig)
            labels.append(f"ID_{pid}")
            
            data_csv.append({"id": pid, "orig": round(s_orig, 4), "pos": round(s_pos, 4), "neg": round(s_neg, 4), "d_pos": round(s_pos-s_orig, 4), "d_neg": round(s_neg-s_orig, 4)})

        if not self.train_mode:
            self._create_plot(scores_orig, labels, "Neutral Scores", "orig")
            self._create_plot(deltas_pos, labels, "Delta Positive", "dpos", (-2, 2))
            self._create_plot(deltas_neg, labels, "Delta Negative", "dneg", (-2, 2))
        
        if self.output_dir:
            pd.DataFrame(data_csv).to_csv(os.path.join(self.output_dir, self.csv_filename), sep=';', index=False)
            print("✅ CSV & Plots Saved.")
    
    @classmethod
    def run(cls, audio_folder, output_dir, csv_filename, train_mode=False, num_samples=20):
        evaluator = cls(audio_folder, output_dir, csv_filename, train_mode)
        evaluator.run_eval(num=num_samples)
        return evaluator
    

# ==========================================
# 7. LAYER ANALYZER
# ==========================================
class LayerAnalyzer:
    def __init__(self, model_wrapper):
        self.mg = model_wrapper

    def run_analysis(self, csv_path, output_dir, layers=None, safe_zone=(15, 38)):
        """
        Esegue l'analisi completa: Estrazione -> Silhouette -> Scelta Best Layer -> PCA.
        
        Args:
            csv_path (str): Path al CSV dataset.
            output_dir (str): Cartella dove salvare i grafici.
            layers (list): Lista di layer da scansionare (default: 0-47).
            safe_zone (tuple): (min, max) Layer range considerati validi per lo steering.
                               I layer fuori da questo range vengono analizzati ma NON scelti come 'Best'.
        """
        if layers is None:
            # Default per MusicGen Melody (48 layers)
            layers = list(range(0, 48))
        
        print(f"🔬 Layer Analysis started on {len(layers)} layers...")
        print(f"🛡️ Safe Zone for selection: Layer {safe_zone[0]} to {safe_zone[1]}")
        os.makedirs(output_dir, exist_ok=True)

        # 1. SETUP HOOKS
        hooks = {}
        for idx in layers:
            try:
                layer_module = self.mg.model.lm.transformer.layers[idx]
                h = ActivationHook(layer_module)
                h.register()
                hooks[idx] = h
            except:
                print(f"⚠️ Warning: Could not hook layer {idx}")

        # 2. DATA LOAD & EXTRACTION
        try:
            df = pd.read_csv(csv_path, sep=';')
            if len(df) > 25: df = df.head(25) # Limitiamo per velocità
        except Exception as e:
            print(f"❌ CSV Error: {e}"); return None

        layer_data = {idx: {'pos': [], 'neg': []} for idx in layers}

        print("   Extracting activations...")
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                p_pos = str(row['positive_prompt']).strip()
                p_neg = str(row['negative_prompt']).strip()
                
                # Gestione Melodia
                melody = row.get('melody_path', None)
                if pd.isna(melody) or isinstance(melody, float): melody = None
                elif isinstance(melody, str) and not os.path.exists(melody): melody = None

                # Generate Happy
                self.mg.generate(p_pos, melody_path=melody)
                for idx, h in hooks.items():
                    vec = h.get_mean_vector().cpu().numpy()
                    layer_data[idx]['pos'].append(vec)
                    h.activations = [] # Reset

                # Generate Sad
                self.mg.generate(p_neg, melody_path=melody)
                for idx, h in hooks.items():
                    vec = h.get_mean_vector().cpu().numpy()
                    layer_data[idx]['neg'].append(vec)
                    h.activations = [] # Reset
            except Exception as e:
                print(f"   Skip row {i}: {e}")

        # Rimuovi Hooks
        for h in hooks.values(): h.remove()

        # 3. COMPUTE SCORES
        print("\n🧮 Computing Silhouette Scores...")
        final_scores = []
        valid_layers_for_selection = []
        valid_scores_for_selection = []

        for idx in layers:
            pos_vecs = np.array(layer_data[idx]['pos'])
            neg_vecs = np.array(layer_data[idx]['neg'])
            
            if len(pos_vecs) == 0: 
                final_scores.append(0)
                continue

            # Flattening dimensions if needed
            if len(pos_vecs.shape) > 2: pos_vecs = pos_vecs.mean(axis=1)
            if len(neg_vecs.shape) > 2: neg_vecs = neg_vecs.mean(axis=1)

            X = np.concatenate([pos_vecs, neg_vecs])
            y = np.array([1]*len(pos_vecs) + [0]*len(neg_vecs))

            score = silhouette_score(X, y)
            final_scores.append(score)
            
            # Filtro per la selezione del "Best Layer" (solo safe zone)
            if safe_zone[0] <= idx <= safe_zone[1]:
                valid_layers_for_selection.append(idx)
                valid_scores_for_selection.append(score)

        # 4. FIND BEST LAYER (IN SAFE ZONE)
        if valid_layers_for_selection:
            best_score_idx = np.argmax(valid_scores_for_selection)
            best_layer = valid_layers_for_selection[best_score_idx]
            best_score_val = valid_scores_for_selection[best_score_idx]
            print(f"🏆 Best Layer (in Safe Zone): {best_layer} (Score: {best_score_val:.4f})")
        else:
            print("⚠️ No valid layers in safe zone found. Using global max.")
            best_layer = layers[np.argmax(final_scores)]

        # 5. PLOT 1: SILHOUETTE SCORE (Separation Analysis)
        plt.figure(figsize=(12, 6))
        plt.plot(layers, final_scores, marker='o', linestyle='-', color='purple', linewidth=2, label='Clustering Score')
        plt.fill_between(layers, final_scores, alpha=0.2, color='purple')
        
        # Evidenzia il Safe Peak
        plt.axvline(x=best_layer, color='red', linestyle='--', linewidth=2, label=f'Chosen Peak (Layer {best_layer})')
        
        # Evidenzia la Safe Zone
        plt.axvspan(safe_zone[0], safe_zone[1], color='green', alpha=0.1, label='Safe Semantic Block')

        plt.title("Emotion Separation Analysis by Layer", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Silhouette Score (Higher is Better)", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plot1_path = os.path.join(output_dir, "analysis_1_separation_scores.png")
        plt.savefig(plot1_path, dpi=300)
        print(f"✅ Saved Score Plot: {plot1_path}")
        plt.close()

        # 6. PLOT 2: PCA VISUALIZATION (For the Best Layer)
        print(f"🎨 Generating PCA for Layer {best_layer}...")
        
        # Retrieve vectors for best layer
        pos_vecs = np.array(layer_data[best_layer]['pos'])
        neg_vecs = np.array(layer_data[best_layer]['neg'])
        
        if len(pos_vecs.shape) > 2: pos_vecs = pos_vecs.mean(axis=1)
        if len(neg_vecs.shape) > 2: neg_vecs = neg_vecs.mean(axis=1)

        X_pca = np.concatenate([pos_vecs, neg_vecs])
        
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_pca)
        
        # Split back
        pos_2d = X_2d[:len(pos_vecs)]
        neg_2d = X_2d[len(pos_vecs):]

        plt.figure(figsize=(10, 8))
        plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c='dodgerblue', s=100, alpha=0.7, label='Happy Prompts', edgecolors='white')
        plt.scatter(neg_2d[:, 0], neg_2d[:, 1], c='crimson', s=100, alpha=0.7, label='Sad Prompts', edgecolors='white')
        
        plt.title(f"Latent Space Visualization (Layer {best_layer})", fontsize=14)
        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plot2_path = os.path.join(output_dir, f"analysis_2_pca_layer_{best_layer}.png")
        plt.savefig(plot2_path, dpi=300)
        print(f"✅ Saved PCA Plot: {plot2_path}")
        plt.close()

        return best_layer