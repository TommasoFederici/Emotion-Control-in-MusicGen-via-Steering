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
import librosa
import warnings

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
        
        # Normalizzazione: Assicuriamoci che il vettore sia unitario (Norma = 1)
        self.vector = steering_vector.to(device).float()
        self.vector = self.vector / (self.vector.norm() + 1e-8)

        # Broadcasting
        if self.vector.dim() == 1: self.vector = self.vector.view(1, 1, -1)
        elif self.vector.dim() == 2: self.vector = self.vector.unsqueeze(1)

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple): h, other = output[0], output[1:]
        else: h, other = output, ()
        
        # Calcolo Alpha
        decayed_value = self.base_alpha * (self.decay ** self.step)
        
        # Gestione segno: se base_alpha è positivo, stiamo sopra min_alpha
        # se è negativo (steering opposto), stiamo sotto -min_alpha
        if self.base_alpha >= 0:
            current_alpha = max(decayed_value, self.min_alpha)
        else:
            current_alpha = min(decayed_value, -self.min_alpha)
            
        self.step += 1
        
        # Applicazione Steering
        h_new = h + (current_alpha * self.vector)
        
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
# 6. EVALUATION CLASS
# ==========================================
class Evaluation:
    def __init__(self, audio_folder, output_dir, csv_filename, train_mode=False, label_pos="happy mood", label_neg="sad mood"):
        """
        Inizializza la classe Evaluation.
        """
        self.audio_folder = audio_folder
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        self.train_mode = train_mode
        
        self.label_pos = label_pos
        self.label_neg = label_neg
        self.candidate_labels = [self.label_pos, self.label_neg]
        self.ids = []

        print(f"--- Inizializzazione Evaluation (Mode: {'TRAIN' if train_mode else 'FULL EVAL'}) ---")
        
        # Scansione ID
        scan_suffix = "_pos.wav" if self.train_mode else "_orig.wav"
        
        try:
            files = os.listdir(self.audio_folder)
            for f in files:
                if f.endswith(scan_suffix): 
                    try:
                        self.ids.append(int(f.split('_')[0]))
                    except ValueError: pass
            self.ids.sort()
            print(f"File trovati (suffix '{scan_suffix}'): {len(self.ids)} ID unici.")
        except FileNotFoundError:
            print(f"ERRORE: Cartella {self.audio_folder} non trovata.")

        
        try:
            self.classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
            print("Modello CLAP caricato.\n")
        except:
            print("⚠️ Errore caricamento CLAP (forse manca libreria?)")

    def _get_valence_score(self, audio_path):
        """Calcola score AI (-1 a +1)"""
        try:
            output = self.classifier(audio_path, candidate_labels=self.candidate_labels)
            score_pos = next(item['score'] for item in output if item['label'] == self.label_pos)
            score_neg = next(item['score'] for item in output if item['label'] == self.label_neg)
            return score_pos - score_neg
        except Exception as e:
            return 0.0

    def extract_acoustic_features(self, audio_path):
        """
        Estrae metriche fisiche dal file audio usando Librosa.
        Ritorna un dizionario con i valori medi: CENTROIDE e BPM.
        """
        if not os.path.exists(audio_path):
            return {"centroid": 0, "bpm": 0}

        try:
            # Carica audio (solo primi 10s per velocità)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(audio_path, duration=10)
            
            # 1. Spectral Centroid (Brillantezza)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            avg_centroid = np.mean(cent)

            # 2. BPM (Tempo) - NUOVO CODICE
            # beat_track ritorna (tempo, beats). A noi serve solo tempo.
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # A volte librosa ritorna un array di 1 elemento, a volte un float. Normalizziamo.
            if isinstance(tempo, np.ndarray):
                tempo = tempo.item()
            
            return {
                "centroid": round(float(avg_centroid), 2),
                "bpm": round(float(tempo), 2)
            }
            
        except Exception as e:
            # In caso di errore (es. file troppo breve o silenzioso)
            return {"centroid": 0, "bpm": 0}

    def _create_bar_chart(self, x_labels, values, title, y_label, y_limit=(-1.1, 1.1)):
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(x_labels)) 
        x_labels_str = [str(x) for x in x_labels]
        
        cmap = plt.get_cmap('bwr')
        norm = mcolors.Normalize(vmin=-1, vmax=1) 
        bar_colors = cmap(norm(values))

        bars = plt.bar(x_pos, values, color=bar_colors, edgecolor='black', width=0.6)
        
        plt.axhline(0, color='black', linewidth=1.5)
        plt.ylim(y_limit[0], y_limit[1])
        
        plt.title(title, fontsize=16)
        
        plt.xlabel("Tracks ID", fontsize=12)  
        plt.ylabel(y_label, fontsize=12)      

        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.xticks(x_pos, x_labels_str, rotation=0) 

        y_range = y_limit[1] - y_limit[0]
        offset = y_range * 0.02

        for bar, score in zip(bars, values):
            y_pos = score + offset if score > 0 else score - (offset * 2)
            plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{score:.2f}", 
                     ha='center', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        
        if self.output_dir:
            safe_name = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
            save_path = os.path.join(self.output_dir, f"plot_{safe_name}.png")
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"📊 Grafico salvato: {save_path}")
        
        plt.show()
        plt.close()

    # --- PLOTTING FUNCTIONS ---
    def plot_neutral(self, num_samples=20, y_label="Score"):
        self._run_single_plot(num_samples, "_orig.wav", "Neutral Analysis", y_label)

    def plot_positive(self, num_samples=20, y_label="Score"):
        self._run_single_plot(num_samples, "_pos.wav", "Positive Analysis", y_label)

    def plot_negative(self, num_samples=20, y_label="Score"):
        self._run_single_plot(num_samples, "_neg.wav", "Negative Analysis", y_label)

    def _run_single_plot(self, num_samples, suffix, title, y_label):
        scores, labels = [], []
        for audio_id in self.ids[:num_samples]:
            path = os.path.join(self.audio_folder, f"{audio_id}{suffix}")
            if os.path.exists(path):
                scores.append(self._get_valence_score(path))
            else:
                scores.append(0)
            labels.append(audio_id)
        self._create_bar_chart(labels, scores, title, y_label)

    def plot_delta_positive(self, num_samples=20, y_label="Score"):
        self._run_delta_plot(num_samples, "_pos.wav", "Delta Positive (Pos - Orig)", y_label)

    def plot_delta_negative(self, num_samples=20, y_label="Score"):
        self._run_delta_plot(num_samples, "_neg.wav", "Delta Negative (Neg - Orig)", y_label)

    def _run_delta_plot(self, num_samples, target_suffix, title, y_label):
        deltas, labels = [], []
        for audio_id in self.ids[:num_samples]:
            path_orig = os.path.join(self.audio_folder, f"{audio_id}_orig.wav")
            path_target = os.path.join(self.audio_folder, f"{audio_id}{target_suffix}")
            
            if os.path.exists(path_orig) and os.path.exists(path_target):
                s_orig = self._get_valence_score(path_orig)
                s_target = self._get_valence_score(path_target)
                deltas.append(s_target - s_orig)
            else:
                deltas.append(0)
            labels.append(audio_id)
        
        self._create_bar_chart(labels, deltas, title, y_label, y_limit=(-2.0, 2.0))

    # --- SALVATAGGIO CSV  ---
    def save_to_csv(self):
        """Salva il CSV includendo metriche fisiche (CENTROIDE E BPM) e riga AVG finale."""
        os.makedirs(self.output_dir, exist_ok=True)
        full_path = os.path.join(self.output_dir, self.csv_filename)

        print(f"\n--- Inizio calcolo features e salvataggio CSV su {len(self.ids)} file ---")
        
        data = []

        for i, audio_id in enumerate(self.ids):
            path_pos = os.path.join(self.audio_folder, f"{audio_id}_pos.wav")
            path_neg = os.path.join(self.audio_folder, f"{audio_id}_neg.wav")
            
            s_pos = self._get_valence_score(path_pos) if os.path.exists(path_pos) else 0.0
            s_neg = self._get_valence_score(path_neg) if os.path.exists(path_neg) else 0.0

            # Estrai features (ora include anche BPM)
            feat_pos = self.extract_acoustic_features(path_pos)
            feat_neg = self.extract_acoustic_features(path_neg)

            if self.train_mode:
                data.append({
                    "id": audio_id,
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4),
                    "pos_centroid": feat_pos['centroid'],
                    "neg_centroid": feat_neg['centroid'],
                    "pos_bpm": feat_pos['bpm'],
                    "neg_bpm": feat_neg['bpm']
                })
            else:
                path_orig = os.path.join(self.audio_folder, f"{audio_id}_orig.wav")
                s_neutral = self._get_valence_score(path_orig) if os.path.exists(path_orig) else 0.0
                feat_orig = self.extract_acoustic_features(path_orig)
                
                data.append({
                    "id": audio_id,
                    "score_neutral": round(s_neutral, 4),
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4),
                    "delta_pos": round(s_pos - s_neutral, 4),
                    "delta_neg": round(s_neg - s_neutral, 4),
                    "orig_centroid": feat_orig['centroid'],
                    "pos_centroid": feat_pos['centroid'],
                    "neg_centroid": feat_neg['centroid'],
                    "orig_bpm": feat_orig['bpm'],
                    "pos_bpm": feat_pos['bpm'],
                    "neg_bpm": feat_neg['bpm']
                })

            if i % 5 == 0:
                print(f"Processati {i}/{len(self.ids)} file...")

        # Creazione DataFrame
        df = pd.DataFrame(data)
        
        # --- CALCOLO E STAMPA MEDIA ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols: numeric_cols.remove('id')
        
        avg_row = {col: round(df[col].mean(), 4) for col in numeric_cols}
        
        # === STAMPA MEDIE ===
        print("\n" + "="*40)
        print("📊  MEDIA PUNTEGGI (AVERAGE SCORES & METRICS)")
        print("="*40)
        if 'score_pos' in avg_row:
            print(f"🔹 Positive Score Avg: {avg_row['score_pos']}")
        if 'score_neg' in avg_row:
            print(f"🔸 Negative Score Avg: {avg_row['score_neg']}")
        
        if 'pos_bpm' in avg_row and 'neg_bpm' in avg_row:
            print(f"🥁 Positive BPM Avg: {avg_row['pos_bpm']}")
            print(f"🥁 Negative BPM Avg: {avg_row['neg_bpm']}")

        # Stampa anche i delta se siamo in FULL MODE
        if 'delta_pos' in avg_row:
            print(f"🔺 Delta Positive Avg: {avg_row['delta_pos']}")
        if 'delta_neg' in avg_row:
            print(f"🔻 Delta Negative Avg: {avg_row['delta_neg']}")
        print("="*40 + "\n")
        # =====================

        avg_row['id'] = "AVG" 
        df_avg = pd.DataFrame([avg_row])
        df = pd.concat([df, df_avg], ignore_index=True)
        
        try:
            df.to_csv(full_path, sep=';', index=False)
            print(f"✅ CSV completato e salvato in: {full_path}")
        except Exception as e:
            print(f"Errore nel salvare il CSV: {e}")

    # --- ORCHESTRATORE ---
    def evaluate(self, num_samples_plot=20, y_label="Score"):
        if self.train_mode:
            print("\n>>> TRAIN MODE <<<")
            self.plot_positive(num_samples_plot, y_label=y_label)
            self.plot_negative(num_samples_plot, y_label=y_label)
        else:
            print("\n>>> FULL MODE <<<")
            self.plot_neutral(num_samples_plot, y_label=y_label)
            self.plot_positive(num_samples_plot, y_label=y_label)
            self.plot_negative(num_samples_plot, y_label=y_label)
            self.plot_delta_positive(num_samples_plot, y_label=y_label)
            self.plot_delta_negative(num_samples_plot, y_label=y_label)
        self.save_to_csv()

    @classmethod
    def run(cls, audio_folder, output_dir, csv_filename, train_mode=False, num_samples=20, y_label="Score", label_pos="happy mood", label_neg="sad mood"):
        evaluator = cls(audio_folder, output_dir, csv_filename, train_mode, label_pos, label_neg)
        evaluator.evaluate(num_samples_plot=num_samples, y_label=y_label)
        return evaluator