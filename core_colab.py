import torch
import pandas as pd
import os
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import pipeline
import numpy as np

# Silenzia log di sistema
logging.getLogger("audiocraft").setLevel(logging.ERROR)

# ==========================================
# 1. WRAPPER MODELLO
# ==========================================
class MusicGenWrapper:
    def __init__(self, size='small', duration=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 MusicGen ({size}) | Device: {self.device} | Durata: {duration}s")
        self.model = MusicGen.get_pretrained(size)
        self.model.set_generation_params(duration=duration)

    def generate(self, prompt, filename=None):
        wav = self.model.generate([prompt], progress=False)
        if filename:
            if filename.endswith(".wav"): filename = filename[:-4]
            # Assicura che la cartella esista
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            audio_write(filename, wav[0].cpu(), self.model.sample_rate, strategy="loudness", loudness_headroom_db=16)
        return wav

# ==========================================
# 2. ACTIVATION HOOK (Per Estrazione)
# ==========================================
class ActivationHook:
    def __init__(self, module):
        self.module = module
        self.handle = None
        self.activations = [] 

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple): output = output[0]
        
        # FIX CFG: Se c'è batch > 1 (Condizionato + Incondizionato), 
        # prendiamo solo la parte condizionata (Indice 0)
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
        # Media sul tempo [Batch, Time, Dim] -> [Dim]
        return full.mean(dim=1).squeeze()

# ==========================================
# 3. DYNAMIC STEERING (Metodo Paper) 🔥
# ==========================================
class DynamicSteering:
    """
    Applica Activation Steering usando la formula normalizzata del paper.
    Formula: h_new = (h_old + alpha * vector) / (1 + alpha)
    """
    def __init__(self, module, steering_vector):
        self.module = module
        self.handle = None
        self.alpha = 0.0
        
        # Setup Device e Vettore
        try:
            device = next(module.parameters()).device
        except:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Il vettore deve essere [1, 1, Dim] per essere sommato a [Batch, Time, Dim]
        self.vector = steering_vector.to(device).float()
        if self.vector.dim() == 1:
            self.vector = self.vector.view(1, 1, -1)
        elif self.vector.dim() == 2: # Se è [1, 1024]
            self.vector = self.vector.unsqueeze(1) # Diventa [1, 1, 1024]

    def hook_fn(self, module, input, output):
        # Gestione output MusicGen (Tuple: Tensor, Cache)
        if isinstance(output, tuple):
            h = output[0]
            other = output[1:]
        else:
            h = output
            other = ()

        # --- FORMULA DEL PAPER ---
        # Questa formula evita che il volume "esploda" quando sommiamo il vettore.
        # h: Flusso originale
        # v: Vettore Emozione
        # alpha: Intensità
        
        # Nota: Usiamo abs(alpha) al denominatore per stabilità se alpha è negativo
        h_new = (h + (self.alpha * self.vector)) / (1 + abs(self.alpha))
        
        if other:
            return (h_new,) + other
        return h_new

    def apply(self, coefficient=1.0):
        """Attiva l'hook con un certo coefficiente."""
        self.alpha = coefficient
        if self.handle is None:
            self.handle = self.module.register_forward_hook(self.hook_fn)
        
    def remove(self):
        """Rimuove l'hook."""
        if self.handle:
            self.handle.remove()
            self.handle = None

# ==========================================
# 4. DATASET EXTRACTOR
# ==========================================
class DatasetExtractor:
    def __init__(self, model_wrapper, layer_idx=14):
        self.mg = model_wrapper
        self.layer_idx = layer_idx
        self.target_layer = self.mg.model.lm.transformer.layers[layer_idx]
        self.hook = ActivationHook(self.target_layer)
        
    def extract(self, csv_path, save_path, audio_output_dir=None, sep=';'):
        print(f"🏭 Extracting from {csv_path}...")
        try: df = pd.read_csv(csv_path, sep=sep)
        except: print("❌ Error reading CSV"); return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if audio_output_dir: os.makedirs(audio_output_dir, exist_ok=True)

        cumulative_vector = None
        count = 0
        self.hook.register()

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                p_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                p_neg = str(row.get('negative_prompt', row.iloc[1])).strip()
                
                # --- FIX: Usa ID dal CSV se c'è, altrimenti usa index ---
                pair_id = str(row.get('ID', index)).strip()

                # Nomi file allineati con Evaluation: {ID}_pos.wav
                f_pos = os.path.join(audio_output_dir, f"{pair_id}_pos") if audio_output_dir else None
                f_neg = os.path.join(audio_output_dir, f"{pair_id}_neg") if audio_output_dir else None

                self.mg.generate(p_pos, filename=f_pos)
                vec_pos = self.hook.get_mean_vector()
                self.hook.activations = []

                self.mg.generate(p_neg, filename=f_neg)
                vec_neg = self.hook.get_mean_vector()
                self.hook.activations = []

                if vec_pos is not None and vec_neg is not None:
                    diff = vec_pos - vec_neg
                    diff = diff / (diff.norm() + 1e-8) # Normalizzazione Locale
                    if cumulative_vector is None: cumulative_vector = diff
                    else: cumulative_vector += diff
                    count += 1
            except Exception as e: print(f"Err row {index}: {e}")

        self.hook.remove()
        if cumulative_vector is not None:
            mean_vector = cumulative_vector / count
            mean_vector = mean_vector / mean_vector.norm() # Normalizzazione Finale
            torch.save(mean_vector, save_path)
            print(f"✅ Saved vector: {save_path}")

# ==========================================
# 5. DATASET INFERENCE (Aggiornata per Dynamic)
# ==========================================
class DatasetInference:
    def __init__(self, model_wrapper, layer_idx=14):
        self.mg = model_wrapper
        self.layer_idx = layer_idx
        self.target_layer = self.mg.model.lm.transformer.layers[layer_idx]

    def run(self, prompts_file, vector_path, output_dir, alpha=1.5, max_samples=None):
        print(f"🚀 Batch Inference (Dynamic Alpha {alpha})...")
        
        try:
            vec = torch.load(vector_path)
            if vec.dim() == 2 and vec.shape[0] > 1: vec = vec.mean(dim=0, keepdim=True)
            vec = vec / (vec.norm() + 1e-8)
        except: print("❌ Error loading vector"); return

        try:
            df = pd.read_csv(prompts_file, sep=';')
            if max_samples: df = df.head(max_samples)
        except: print("❌ Error reading CSV"); return

        os.makedirs(output_dir, exist_ok=True)
        
        # Usiamo DynamicSteering invece di WeightSteering
        steerer = DynamicSteering(self.target_layer, vec)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                pid = str(row.get('ID', row.get('id', i))).strip()
                if 'test_prompt' in row: prompt = str(row['test_prompt']).strip()
                elif 'prompt' in row: prompt = str(row['prompt']).strip()
                else: prompt = str(row.iloc[-1]).strip()
                
                base = os.path.join(output_dir, f"{pid}")

                # Genera Originale
                self.mg.generate(prompt, f"{base}_orig")
                
                # Genera Happy
                steerer.apply(alpha)
                self.mg.generate(prompt, f"{base}_pos")
                steerer.remove()
                
                # Genera Sad (Alpha negativo per inversione)
                # Attenzione: con la formula (1+alpha), usare alpha negativi può essere tricky
                # se alpha è vicino a -1 (divisione per zero).
                # La nostra classe usa abs(alpha) al denominatore, quindi è sicuro.
                steerer.apply(-alpha)
                self.mg.generate(prompt, f"{base}_neg")
                steerer.remove()
                
            except Exception as e:
                print(f"Error {pid}: {e}")
                steerer.remove()


# ==========================================
# 6. EVALUATION CLASS
# ==========================================
class Evaluation:
    def __init__(self, audio_folder, output_dir, csv_filename, train_mode=False, label_pos="happy mood", label_neg="sad mood"):
        """
        Inizializza la classe Evaluation.
        Args:
            audio_folder (str): Cartella con i file audio.
            output_dir (str): Cartella dove salvare il CSV (e ora i grafici).
            csv_filename (str): Nome del file CSV finale.
            train_mode (bool): Se True, calcola solo Pos/Neg. Se False, calcola anche Orig/Delta.
            label_pos (str): Etichetta positiva.
            label_neg (str): Etichetta negativa.
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
        """Calcola score (-1 a +1)"""
        try:
            output = self.classifier(audio_path, candidate_labels=self.candidate_labels)
            score_pos = next(item['score'] for item in output if item['label'] == self.label_pos)
            score_neg = next(item['score'] for item in output if item['label'] == self.label_neg)
            return score_pos - score_neg
        except Exception as e:
            return 0.0

    def _create_bar_chart(self, x_labels, values, title, y_limit=(-1.1, 1.1)):
        plt.figure(figsize=(12, 6))
        cmap = plt.get_cmap('bwr')
        norm = mcolors.Normalize(vmin=-1, vmax=1) 
        bar_colors = cmap(norm(values))

        bars = plt.bar(x_labels, values, color=bar_colors, edgecolor='black', width=0.6)
        plt.axhline(0, color='black', linewidth=1.5)
        plt.ylim(y_limit[0], y_limit[1])
        plt.title(title, fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)

        y_range = y_limit[1] - y_limit[0]
        offset = y_range * 0.02

        for bar, score in zip(bars, values):
            y_pos = score + offset if score > 0 else score - (offset * 2)
            plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{score:.2f}", 
                     ha='center', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        
        # --- MODIFICA: SALVATAGGIO GRAFICO ---
        if self.output_dir:
            # Crea nome file sicuro dal titolo (es. "Delta Positive" -> "plot_delta_positive.png")
            safe_name = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
            save_path = os.path.join(self.output_dir, f"plot_{safe_name}.png")
            # Crea cartella se non esiste
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"📊 Grafico salvato: {save_path}")
        
        plt.show()
        plt.close()

    # --- PLOTTING FUNCTIONS ---
    def plot_neutral(self, num_samples=20):
        self._run_single_plot(num_samples, "_orig.wav", "Neutral Analysis")

    def plot_positive(self, num_samples=20):
        self._run_single_plot(num_samples, "_pos.wav", "Positive Analysis")

    def plot_negative(self, num_samples=20):
        self._run_single_plot(num_samples, "_neg.wav", "Negative Analysis")

    def _run_single_plot(self, num_samples, suffix, title):
        scores, labels = [], []
        for audio_id in self.ids[:num_samples]:
            path = os.path.join(self.audio_folder, f"{audio_id}{suffix}")
            if os.path.exists(path):
                scores.append(self._get_valence_score(path))
            else:
                scores.append(0)
            labels.append(f"ID_{audio_id}")
        self._create_bar_chart(labels, scores, title)

    def plot_delta_positive(self, num_samples=20):
        self._run_delta_plot(num_samples, "_pos.wav", "Delta Positive (Pos - Orig)")

    def plot_delta_negative(self, num_samples=20):
        self._run_delta_plot(num_samples, "_neg.wav", "Delta Negative (Neg - Orig)")

    def _run_delta_plot(self, num_samples, target_suffix, title):
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
            labels.append(f"ID_{audio_id}")
        
        self._create_bar_chart(labels, deltas, title, y_limit=(-2.0, 2.0))

    # --- SALVATAGGIO CSV ---
    def save_to_csv(self):
        """
        Salva il CSV in base alla modalità (TRAIN o FULL).
        """
        # Assicura che la cartella di output esista
        os.makedirs(self.output_dir, exist_ok=True)
        full_path = os.path.join(self.output_dir, self.csv_filename)

        print(f"\n--- Inizio calcolo e salvataggio CSV su {len(self.ids)} file ---")
        print(f"Modalità TRAIN: {self.train_mode}")
        
        data = []

        for i, audio_id in enumerate(self.ids):
            # Percorsi comuni
            path_pos = os.path.join(self.audio_folder, f"{audio_id}_pos.wav")
            path_neg = os.path.join(self.audio_folder, f"{audio_id}_neg.wav")
            
            # Calcolo score Pos/Neg
            s_pos = self._get_valence_score(path_pos) if os.path.exists(path_pos) else 0.0
            s_neg = self._get_valence_score(path_neg) if os.path.exists(path_neg) else 0.0

            if self.train_mode:
                # MODALITA' TRAIN: Solo Pos e Neg
                data.append({
                    "id": audio_id,
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4)
                })
            else:
                # MODALITA' DEFAULT: Anche Orig e Delta
                path_orig = os.path.join(self.audio_folder, f"{audio_id}_orig.wav")
                s_neutral = self._get_valence_score(path_orig) if os.path.exists(path_orig) else 0.0
                
                d_pos = s_pos - s_neutral
                d_neg = s_neg - s_neutral

                data.append({
                    "id": audio_id,
                    "score_neutral": round(s_neutral, 4),
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4),
                    "delta_pos": round(d_pos, 4),
                    "delta_neg": round(d_neg, 4)
                })

            if i % 5 == 0:
                print(f"Processati {i}/{len(self.ids)} file...")

        # Creazione DataFrame
        df = pd.DataFrame(data)
        
        # Ordinamento colonne
        if self.train_mode:
            cols = ["id", "score_pos", "score_neg"]
        else:
            cols = ["id", "score_neutral", "score_pos", "score_neg", "delta_pos", "delta_neg"]
        
        df = df[cols]
        
        try:
            df.to_csv(full_path, sep=';', index=False)
            print(f"\nCOMPLETATO. File salvato in: {full_path}")
            print(df.head()) 
        except Exception as e:
            print(f"Errore nel salvare il CSV: {e}")

    # --- FUNZIONE ORCHESTRATORE ---
    def evaluate(self, num_samples_plot=20):
        if self.train_mode:
            print("\n>>> TRAIN MODE <<<")
            self.plot_positive(num_samples_plot)
            self.plot_negative(num_samples_plot)
        else:
            print("\n>>> FULL MODE <<<")
            self.plot_neutral(num_samples_plot)
            self.plot_positive(num_samples_plot)
            self.plot_negative(num_samples_plot)
            self.plot_delta_positive(num_samples_plot)
            self.plot_delta_negative(num_samples_plot)
        self.save_to_csv()

    @classmethod
    def run(cls, audio_folder, output_dir, csv_filename, train_mode=False, num_samples=20):
        evaluator = cls(audio_folder, output_dir, csv_filename, train_mode)
        evaluator.evaluate(num_samples_plot=num_samples)
        return evaluator