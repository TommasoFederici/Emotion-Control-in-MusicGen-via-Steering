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
# 3. DYNAMIC STEERING (Identico)
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
        self.default_layers = layers if isinstance(layers, list) else [layers] if layers else [14]

    def run(self, prompts_file, vector_path, output_dir, alpha=1.5, max_samples=None):
        print(f"🚀 Multi-Layer Inference (Alpha {alpha})...")
        
        steerers = []
        try:
            data = torch.load(vector_path)
            if isinstance(data, dict):
                print(f"📦 Multi-Layer: {list(data.keys())}")
                for idx, vec in data.items():
                    target = self.mg.model.lm.transformer.layers[idx]
                    vec = vec / (vec.norm() + 1e-8)
                    if vec.dim() == 2 and vec.shape[0] > 1: vec = vec.mean(dim=0, keepdim=True)
                    steerers.append(DynamicSteering(target, vec))
            else:
                print(f"📦 Single-Layer: {self.default_layers}")
                vec = data
                vec = vec / (vec.norm() + 1e-8)
                for idx in self.default_layers:
                    target = self.mg.model.lm.transformer.layers[idx]
                    steerers.append(DynamicSteering(target, vec))
        except Exception as e: print(f"❌ Vector error: {e}"); return

        if not steerers: print("❌ No steerers."); return

        try:
            df = pd.read_csv(prompts_file, sep=';')
            if max_samples: df = df.head(max_samples)
        except: print("❌ CSV error"); return

        os.makedirs(output_dir, exist_ok=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                pid = str(row.get('ID', row.get('id', i))).strip()
                
                # Parsing Prompt
                if 'test_prompt' in row: prompt = str(row['test_prompt']).strip()
                elif 'prompt' in row: prompt = str(row['prompt']).strip()
                else: prompt = str(row.iloc[-1]).strip()
                
                # Parsing Melody
                melody_file = None
                if 'melody_path' in row and pd.notna(row['melody_path']):
                    melody_file = str(row['melody_path']).strip()

                safe_p = "".join([c for c in prompt if c.isalnum() or c in " _-"])[:20].replace(" ", "_")
                base = os.path.join(output_dir, f"{pid}")

                # A. Originale
                self.mg.generate(prompt, f"{base}_orig", melody_path=melody_file)
                
                # B. Happy
                for s in steerers: s.apply(alpha)
                self.mg.generate(prompt, f"{base}_pos", melody_path=melody_file)
                for s in steerers: s.remove()
                
                # C. Sad
                for s in steerers: s.apply(-alpha)
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