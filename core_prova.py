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

    def generate(self, prompt, filename=None, melody_path=None, use_melody=False):
        """
        Genera audio. 
        Arg 'ignore_melody': Se True, ignora melody_path anche se è presente.
        """
        # Se richiesto, forziamo melody_path a None
        if not use_melody:
            melody_path = None

        if melody_path:
            melody_wav, melody_sr = self.load_melody(melody_path)
            if melody_wav is not None:
                if melody_wav.dim() == 2: melody_wav = melody_wav.unsqueeze(0)
                
                wav = self.model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=melody_wav,
                    melody_sample_rate=melody_sr,
                    progress=False
                )
            else:
                print(f"⚠️ Fallback a testo-solo per '{prompt}'")
                wav = self.model.generate([prompt], progress=False)
        else:
            # Generazione solo testo (Pura, come richiesto dal paper per l'estrazione)
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
# 3. DYNAMIC STEERING (Energy Preserving Fix)
# ==========================================
class DynamicSteering:
    def __init__(self, module, steering_vector):
        self.module = module
        self.handle = None
        self.base_alpha = 0.0
        self.decay = 1.0 
        self.min_alpha = 0.0
        self.step = 0    
        
        try: device = next(module.parameters()).device
        except: device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Normalizzazione preventiva del vettore di steering
        self.vector = steering_vector.to(device).float()
        self.vector = self.vector / (self.vector.norm() + 1e-8)

        # Broadcasting [1, 1, D]
        if self.vector.dim() == 1: self.vector = self.vector.view(1, 1, -1)
        elif self.vector.dim() == 2: self.vector = self.vector.unsqueeze(1)

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple): h, other = output[0], output[1:]
        else: h, other = output, ()
        
        # 1. Calcola l'Energia (Norma) Originale del segnale
        # Questo ci serve per non distruggere il volume/coerenza
        orig_norm = h.norm(p=2, dim=-1, keepdim=True)
        
        # 2. Calcola Alpha
        decayed_value = self.base_alpha * (self.decay ** self.step)
        if self.base_alpha > 0:
            current_alpha = max(decayed_value, self.min_alpha)
        else:
            current_alpha = min(decayed_value, -self.min_alpha)
        self.step += 1
        
        # 3. Applica Steering (Somma Semplice)
        # Qui cambiamo la direzione del vettore
        h_steered = h + (current_alpha * self.vector)
        
        # 4. Rinormalizzazione (Energy Preservation)
        # Riportiamo il vettore sterzato alla stessa magnitudine dell'originale
        # Questo impedisce l'esplosione (rumore) o il collasso (silenzio)
        new_norm = h_steered.norm(p=2, dim=-1, keepdim=True)
        h_new = h_steered * (orig_norm / (new_norm + 1e-8))
        
        if other: return (h_new,) + other
        return h_new

    def apply(self, coefficient=1.0, decay=1.0, min_alpha=0.0):
        self.base_alpha = coefficient
        self.decay = decay
        self.min_alpha = abs(min_alpha)
        self.step = 0 
        if self.handle is None:
            self.handle = self.module.register_forward_hook(self.hook_fn)
            
    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None
        self.step = 0
    
    def reset_steps(self):
        self.step = 0

# ==========================================
# 4. DATASET EXTRACTOR (With Full Analysis)
# ==========================================
class DatasetExtractor:
    def __init__(self, model_wrapper, layers=None):
        self.mg = model_wrapper
        
        # Se layers è None, prendiamo TUTTI i layer disponibili (0-47 per MusicGen Melody)
        if layers is None:
            num_layers = len(self.mg.model.lm.transformer.layers)
            self.target_layers_indices = list(range(num_layers))
            print(f"🔬 Configured for FULL SCAN on {num_layers} layers.")
        else:
            self.target_layers_indices = layers if isinstance(layers, list) else [layers]
            print(f"🔬 Configured for selective scan on {len(self.target_layers_indices)} layers.")
        
        # Setup Hooks
        self.hooks = {}
        for idx in self.target_layers_indices:
            try:
                layer_module = self.mg.model.lm.transformer.layers[idx]
                self.hooks[idx] = ActivationHook(layer_module)
            except IndexError:
                print(f"❌ ERRORE: Layer {idx} non esiste (saltato).")

    def run(self, csv_path, save_vector_path, output_dir_analysis, 
            audio_output_dir=None, safe_zone=(10, 20), use_melody=False):
        """
        Esegue estrazione su TUTTI i layer configurati, calcola i vettori e analizza i risultati.
        Salva un dizionario contenente un vettore per OGNI layer analizzato.
        """
        print(f"🏭 Starting Full Extraction Pipeline...")
        
        try: 
            df = pd.read_csv(csv_path, sep=';')
        except: 
            print("❌ Error reading CSV"); return None

        os.makedirs(os.path.dirname(save_vector_path), exist_ok=True)
        os.makedirs(output_dir_analysis, exist_ok=True)
        if audio_output_dir: os.makedirs(audio_output_dir, exist_ok=True)

        raw_data = {idx: {'pos': [], 'neg': []} for idx in self.hooks.keys()}
        
        # 1. Registra Hooks
        for h in self.hooks.values(): h.register()

        # 2. GENERATION LOOP (Passaggio Unico)
        valid_pairs = 0
        print(f"   Processing {len(df)} prompt pairs...")
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                p_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                p_neg = str(row.get('negative_prompt', row.iloc[1])).strip()
                
                melody_file = None
                if 'melody_path' in row and pd.notna(row['melody_path']):
                    melody_file = str(row['melody_path']).strip()
                
                pid = str(row.get('ID', index)).strip()

                # --- Positive Pass ---
                f_pos = os.path.join(audio_output_dir, f"{pid}_pos") if audio_output_dir else None
                self.mg.generate(p_pos, filename=f_pos, melody_path=melody_file, use_melody=use_melody)
                
                # Collect Pos
                current_pos_vecs = {}
                for idx, h in self.hooks.items():
                    v = h.get_mean_vector()
                    if v is not None: current_pos_vecs[idx] = v.detach().cpu()
                    h.activations = [] 

                # --- Negative Pass ---
                f_neg = os.path.join(audio_output_dir, f"{pid}_neg") if audio_output_dir else None
                self.mg.generate(p_neg, filename=f_neg, melody_path=melody_file, use_melody=use_melody)
                
                # Collect Neg
                current_neg_vecs = {}
                for idx, h in self.hooks.items():
                    v = h.get_mean_vector()
                    if v is not None: current_neg_vecs[idx] = v.detach().cpu()
                    h.activations = [] 

                # Store Pair (Solo se completo)
                pair_complete = True
                for idx in self.hooks.keys():
                    if idx not in current_pos_vecs or idx not in current_neg_vecs:
                        pair_complete = False
                        break
                
                if pair_complete:
                    for idx in self.hooks.keys():
                        raw_data[idx]['pos'].append(current_pos_vecs[idx])
                        raw_data[idx]['neg'].append(current_neg_vecs[idx])
                    valid_pairs += 1

            except Exception as e:
                print(f"   Err row {index}: {e}")

        # Rimuovi Hooks
        for h in self.hooks.values(): h.remove()
        
        if valid_pairs == 0:
            print("❌ Nessun dato valido estratto."); return None

        # 3. CALCOLO VETTORI E ANALISI
        print(f"\n📊 Computing Vectors & Scores for {len(raw_data)} layers...")
        
        final_vectors_dict = {}
        silhouette_scores = []
        layer_indices = []
        
        best_layer_idx = -1
        best_score_val = -100

        # Iteriamo sulle chiavi ordinate di raw_data per sicurezza
        for idx in sorted(raw_data.keys()):
            # Convert lists to tensors
            pos_stack = torch.stack(raw_data[idx]['pos']) # [N, D]
            neg_stack = torch.stack(raw_data[idx]['neg']) # [N, D]
            
            # --- A. Calcolo DiffMean Vector ---
            mean_pos = pos_stack.mean(dim=0)
            mean_neg = neg_stack.mean(dim=0)
            diff_vec = mean_pos - mean_neg
            
            # Normalizzazione
            diff_vec = diff_vec / (diff_vec.norm() + 1e-8)
            final_vectors_dict[idx] = diff_vec

            # --- B. Analisi Silhouette ---
            X_pos = pos_stack.numpy()
            X_neg = neg_stack.numpy()
            
            if X_pos.ndim > 2: X_pos = X_pos.mean(axis=2)
            if X_neg.ndim > 2: X_neg = X_neg.mean(axis=2)
            
            X = np.concatenate([X_pos, X_neg], axis=0)
            y = np.array([1]*len(X_pos) + [0]*len(X_neg))
            
            try:
                score = silhouette_score(X, y)
            except:
                score = 0.0
            
            silhouette_scores.append(score)
            layer_indices.append(idx)
            
            if safe_zone[0] <= idx <= safe_zone[1]:
                if score > best_score_val:
                    best_score_val = score
                    best_layer_idx = idx

        if best_layer_idx == -1 and len(silhouette_scores) > 0:
            best_idx_loc = np.argmax(silhouette_scores)
            best_layer_idx = layer_indices[best_idx_loc]

        print(f"💾 Saving vectors for ALL {len(final_vectors_dict)} layers to {save_vector_path}")
        torch.save(final_vectors_dict, save_vector_path)
        
        print(f"🏆 Suggested 'Best Layer' (in safe zone {safe_zone}): {best_layer_idx} (Score: {best_score_val:.4f})")

        # 4. PLOTTING
        plt.figure(figsize=(12, 6))
        plt.plot(layer_indices, silhouette_scores, marker='o', color='purple', linewidth=2)
        plt.axvline(x=best_layer_idx, color='red', linestyle='--', label=f'Best ({best_layer_idx})')
        plt.axvspan(safe_zone[0], safe_zone[1], color='green', alpha=0.1, label='Target Zone')
        plt.title("Semantic Separation Score across ALL Layers")
        plt.xlabel("Layer Index"); plt.ylabel("Silhouette Score")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir_analysis, "full_model_analysis_scores.png"))
        plt.close()

        if best_layer_idx in raw_data:
            self._plot_pca(raw_data[best_layer_idx], best_layer_idx, output_dir_analysis)


    def _plot_pca(self, layer_raw_data, layer_idx, output_dir):
        try:
            pos_vecs = torch.stack(layer_raw_data['pos']).numpy()
            neg_vecs = torch.stack(layer_raw_data['neg']).numpy()
            
            if pos_vecs.ndim > 2: pos_vecs = pos_vecs.mean(axis=2)
            if neg_vecs.ndim > 2: neg_vecs = neg_vecs.mean(axis=2)

            pca = PCA(n_components=2)
            X_all = np.concatenate([pos_vecs, neg_vecs])
            X_pca = pca.fit_transform(X_all)
            
            half = len(pos_vecs)
            plt.figure(figsize=(8, 8))
            plt.scatter(X_pca[:half, 0], X_pca[:half, 1], c='crimson', label='Pos', alpha=0.7)
            plt.scatter(X_pca[half:, 0], X_pca[half:, 1], c='dodgerblue', label='Neg', alpha=0.7)
            plt.title(f"PCA Latent Space - Layer {layer_idx}")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"pca_layer_{layer_idx}.png"))
            plt.close()
        except Exception as e:
            print(f"⚠️ PCA Plot failed: {e}")


# ==========================================
# 5. DATASET INFERENCE (Con Supporto Melodia)
# ==========================================
class DatasetInference:
    def __init__(self, model_wrapper):
        self.mg = model_wrapper

    def run(self, prompts_file, vector_path, output_dir, 
            injection_map=None, # NUOVO: Dizionario {source_layer: [target_layers]}
            alpha=1.5, decay=1.0, min_alpha=0.0, 
            max_samples=None,
            use_melody=False):
        
        """
        injection_map: Dizionario che definisce la strategia.
           Esempio Multi-Blocco:
           {
             14: [10, 11, 12, 13, 14, 15, 16, 17, 18],  # Vettore del 14 applicato al blocco 10-18
             32: [30, 31, 32, 33, 34, 35]               # Vettore del 32 applicato al blocco 30-35
           }
        """

        print(f"🚀 Multi-Layer Inference (Multi-Block Strategy)...")
        
        steerers = []
        try:
            data = torch.load(vector_path)
            
            # Se l'utente non passa una mappa, usiamo il default (All-to-All o quello che c'è nel file)
            if injection_map is None:
                print("⚠️ No injection_map provided. Fallback to All-to-All matching.")
                if isinstance(data, dict):
                    injection_map = {k: [k] for k in data.keys()} # 14->14, 15->15...
                else:
                    injection_map = {14: [14]} # Default fallback

            # Ciclo sulla mappa di iniezione
            for source_idx, target_list in injection_map.items():
                
                # 1. Recupera il vettore SORGENTE
                if isinstance(data, dict):
                    if source_idx not in data:
                        print(f"⚠️ Warning: Source vector {source_idx} not found. Skipping block.")
                        continue
                    vec = data[source_idx]
                else:
                    vec = data # Caso file con un solo vettore
                
                # Pre-processing vettore
                vec = vec / (vec.norm() + 1e-8)
                if vec.dim() == 2 and vec.shape[0] > 1: vec = vec.mean(dim=0, keepdim=True)
                
                print(f"   🔌 Source Layer {source_idx} -> Injecting into {target_list}")

                # 2. Crea Steerer per ogni layer TARGET
                for target_idx in target_list:
                    if target_idx >= len(self.mg.model.lm.transformer.layers):
                        continue # Evita crash se layer non esiste
                        
                    target_module = self.mg.model.lm.transformer.layers[target_idx]
                    
                    # Gestione Alpha (se vuoi alpha diversi per blocco, puoi passare alpha come dict qui)
                    current_alpha = alpha
                    if isinstance(alpha, dict):
                        # Esempio: se alpha={'mid': 1.5, 'deep': 0.8}
                        if target_idx < 20: current_alpha = alpha.get('mid', 1.5)
                        else: current_alpha = alpha.get('deep', 1.5)
                    
                    s = DynamicSteering(target_module, vec)
                    s.target_alpha = float(current_alpha)
                    s.target_decay = decay
                    s.target_min = min_alpha
                    steerers.append(s)

        except Exception as e: print(f"❌ Setup error: {e}"); return

        if not steerers: print("❌ No steerers active."); return
        print(f"   ✅ Total Steerers Attached: {len(steerers)}")

        # --- CICLO DI INFERENZA (Identico a prima) ---
        try:
            df = pd.read_csv(prompts_file, sep=';')
            if max_samples: df = df.head(max_samples)
        except: print("❌ CSV error"); return

        os.makedirs(output_dir, exist_ok=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Parsing Prompt
                pid = str(row.get('ID', row.get('id', i))).strip()
                if 'test_prompt' in row: prompt = str(row['test_prompt']).strip()
                elif 'prompt' in row: prompt = str(row['prompt']).strip()
                else: prompt = str(row.iloc[-1]).strip()
                
                melody_file = None
                if 'melody_path' in row and pd.notna(row['melody_path']):
                    melody_file = str(row['melody_path']).strip()

                base = os.path.join(output_dir, f"{pid}")

                # A. Originale
                self.mg.generate(prompt, f"{base}_orig", melody_path=melody_file, use_melody=use_melody)
                
                # B. Happy (Positivo)
                for s in steerers: 
                    s.reset_steps()
                    s.apply(s.target_alpha, decay=s.target_decay, min_alpha=s.target_min)
                self.mg.generate(prompt, f"{base}_pos", melody_path=melody_file, use_melody=use_melody)
                for s in steerers: s.remove()
                
                # C. Sad (Negativo)
                for s in steerers: 
                    s.reset_steps()
                    s.apply(-s.target_alpha, decay=s.target_decay, min_alpha=s.target_min)
                self.mg.generate(prompt, f"{base}_neg", melody_path=melody_file, use_melody=use_melody)
                for s in steerers: s.remove()
                
            except Exception as e:
                print(f"Error row {i}: {e}")
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
