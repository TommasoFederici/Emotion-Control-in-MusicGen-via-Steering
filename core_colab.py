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
from sklearn.decomposition import PCA

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
# 4. DATASET EXTRACTOR (multi-layer)
# ==========================================
class DatasetExtractor:
    def __init__(self, model_wrapper, layers=[14]):
        """
        Args:
            layers: Può essere un int (14) o una lista ([12, 13, 14])
        """
        self.mg = model_wrapper
        
        if isinstance(layers, int): layers = [layers]
        self.target_layers_indices = layers
        
        # Dizionario di Hook: {layer_idx: HookInstance}
        self.hooks = {}
        for idx in self.target_layers_indices:
            layer_module = self.mg.model.lm.transformer.layers[idx]
            self.hooks[idx] = ActivationHook(layer_module)
        
    def extract(self, csv_path, save_path, audio_output_dir=None, sep=';'):
        print(f"🏭 Multi-Extraction {self.target_layers_indices} -> {save_path}")
        
        try: df = pd.read_csv(csv_path, sep=sep)
        except: print("❌ Error reading CSV"); return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if audio_output_dir: os.makedirs(audio_output_dir, exist_ok=True)

        # Accumulatori per ogni layer: {14: TensorAccumulatore, 15: ...}
        layer_sums = {idx: None for idx in self.target_layers_indices}
        count = 0

        # Attiva tutti gli hook
        for h in self.hooks.values(): h.register()

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                p_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                p_neg = str(row.get('negative_prompt', row.iloc[1])).strip()
                pid = str(row.get('ID', index)).strip()

                # --- GENERA POSITIVE ---
                f_pos = os.path.join(audio_output_dir, f"{pid}_pos") if audio_output_dir else None
                self.mg.generate(p_pos, filename=f_pos)
                vecs_pos = {idx: h.get_mean_vector() for idx, h in self.hooks.items()}
                for h in self.hooks.values(): h.activations = [] 

                # --- GENERA NEGATIVE ---
                f_neg = os.path.join(audio_output_dir, f"{pid}_neg") if audio_output_dir else None
                self.mg.generate(p_neg, filename=f_neg)
                vecs_neg = {idx: h.get_mean_vector() for idx, h in self.hooks.items()}
                for h in self.hooks.values(): h.activations = [] 

                # --- CALCOLA DIFFERENZE E ACCUMULA ---
                valid_pair = True
                current_diffs = {}
                
                for idx in self.target_layers_indices:
                    v_p = vecs_pos[idx]
                    v_n = vecs_neg[idx]
                    if v_p is None or v_n is None: 
                        valid_pair = False; break
                    
                    diff = v_p - v_n
                    # Normalizzazione Locale (Fondamentale)
                    diff = diff / (diff.norm() + 1e-8)
                    current_diffs[idx] = diff

                if valid_pair:
                    for idx, diff in current_diffs.items():
                        if layer_sums[idx] is None: layer_sums[idx] = diff
                        else: layer_sums[idx] += diff
                    count += 1

            except Exception as e: print(f"Err {index}: {e}")

        # Rimuovi hook
        for h in self.hooks.values(): h.remove()

        # --- CALCOLO VETTORE FINALE PER OGNI LAYER ---
        final_vectors_dict = {}
        
        if count > 0:
            print(f"🧮 Calcolo medie su {count} coppie...")
            for idx, total_sum in layer_sums.items():
                if total_sum is None: continue
                
                # Media Semplice
                mean_vec = total_sum / count
                
                # Normalizzazione Finale
                mean_vec = mean_vec / (mean_vec.norm() + 1e-8)
                if mean_vec.dim() == 1: mean_vec = mean_vec.unsqueeze(0)
                
                final_vectors_dict[idx] = mean_vec

            # Salviamo un DIZIONARIO {layer_idx: vector}
            torch.save(final_vectors_dict, save_path)
            print(f"✅ Multi-Layer Vector salvato: {save_path}")
            print(f"   Contiene layer: {list(final_vectors_dict.keys())}")
        else:
            print("❌ Nessun vettore estratto (count=0).")

        def extract_via_pca(self, csv_path, save_path, audio_output_dir=None, sep=';'):
            print(f"🏭 Multi-Extraction PCA {self.target_layers_indices} -> {save_path}")
            
            try: df = pd.read_csv(csv_path, sep=sep)
            except: print("❌ Error reading CSV"); return

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if audio_output_dir: os.makedirs(audio_output_dir, exist_ok=True)

            # MODIFICA 1: Accumulatori come LISTE, non come somme
            # layer_vectors = {14: [tensor1, tensor2...], 15: [...]}
            layer_vectors = {idx: [] for idx in self.target_layers_indices}
            count = 0

            # Attiva tutti gli hook
            for h in self.hooks.values(): h.register()

            for index, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    p_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                    p_neg = str(row.get('negative_prompt', row.iloc[1])).strip()
                    pid = str(row.get('ID', index)).strip()

                    # --- GENERA POSITIVE ---
                    f_pos = os.path.join(audio_output_dir, f"{pid}_pos") if audio_output_dir else None
                    self.mg.generate(p_pos, filename=f_pos)
                    vecs_pos = {idx: h.get_mean_vector() for idx, h in self.hooks.items()}
                    for h in self.hooks.values(): h.activations = [] 

                    # --- GENERA NEGATIVE ---
                    f_neg = os.path.join(audio_output_dir, f"{pid}_neg") if audio_output_dir else None
                    self.mg.generate(p_neg, filename=f_neg)
                    vecs_neg = {idx: h.get_mean_vector() for idx, h in self.hooks.items()}
                    for h in self.hooks.values(): h.activations = [] 

                    # --- CALCOLA DIFFERENZE E ACCUMULA ---
                    valid_pair = True
                    current_diffs = {}
                    
                    for idx in self.target_layers_indices:
                        v_p = vecs_pos[idx]
                        v_n = vecs_neg[idx]
                        if v_p is None or v_n is None: 
                            valid_pair = False; break
                        
                        diff = v_p - v_n
                        # Normalizzazione Locale
                        diff = diff / (diff.norm() + 1e-8)
                        current_diffs[idx] = diff

                    if valid_pair:
                        for idx, diff in current_diffs.items():
                            # MODIFICA 2: Appendiamo alla lista invece di sommare
                            # Spostiamo su CPU per risparmiare VRAM durante l'accumulo
                            layer_vectors[idx].append(diff.cpu())
                        count += 1

                except Exception as e: print(f"Err {index}: {e}")

            # Rimuovi hook
            for h in self.hooks.values(): h.remove()

            # --- CALCOLO VETTORE FINALE CON PCA ---
            final_vectors_dict = {}
            
            if count > 0:
                print(f"🧮 Calcolo PCA su {count} coppie...")
                
                for idx, vec_list in layer_vectors.items():
                    if not vec_list: continue
                    
                    # 1. Creiamo la matrice [N_samples, Hidden_Dim] (es. 20x1024)
                    matrix = torch.stack(vec_list).numpy()
                    
                    # 2. Applichiamo PCA per trovare la componente principale
                    pca = PCA(n_components=1)
                    pca.fit(matrix)
                    
                    # Il vettore "puro" è la prima componente
                    comp_vec = torch.tensor(pca.components_[0], dtype=torch.float32)
                    
                    # 3. CONTROLLO DIREZIONE (Cruciale)
                    # La PCA può restituire il vettore invertito (negativo).
                    # Calcoliamo la media semplice per capire la direzione generale "Pos - Neg"
                    mean_vec = torch.mean(torch.stack(vec_list), dim=0)
                    
                    # Se il prodotto scalare è negativo, la PCA punta dalla parte opposta -> invertiamo
                    if torch.dot(comp_vec, mean_vec) < 0:
                        comp_vec = -comp_vec
                    
                    # 4. Normalizzazione Finale
                    comp_vec = comp_vec / (comp_vec.norm() + 1e-8)
                    if comp_vec.dim() == 1: comp_vec = comp_vec.unsqueeze(0)
                    
                    final_vectors_dict[idx] = comp_vec
                    
                    # Info Debug: Se Explained Variance è bassa (<0.2), il vettore è molto rumoroso
                    print(f"   Layer {idx}: Variance Explained = {pca.explained_variance_ratio_[0]:.4f}")

                # Salviamo il dizionario
                torch.save(final_vectors_dict, save_path)
                print(f"✅ Multi-Layer PCA Vector salvato: {save_path}")
                print(f"   Contiene layer: {list(final_vectors_dict.keys())}")
            else:
                print("❌ Nessun vettore estratto (count=0).")

# ==========================================
# 5. DATASET INFERENCE (multi-layer)
# ==========================================
class DatasetInference:
    def __init__(self, model_wrapper, layers=None):
        """
        layers: Lista di layer da USARE per l'inferenza.
                Se None, usa TUTTI i layer trovati nel file .pt.
        """
        self.mg = model_wrapper
        # Normalizza a lista
        if isinstance(layers, int): layers = [layers]
        self.filter_layers = layers

    def run(self, prompts_file, vector_path, output_dir, alpha=1.5, max_samples=None):
        print(f"🚀 Batch Inference (Alpha {alpha})...")
        
        steerers = []
        try:
            data = torch.load(vector_path)
            
            # Determina quali layer usare
            available_layers = []
            if isinstance(data, dict):
                available_layers = list(data.keys())
            else:
                available_layers = [14] # Fallback per vecchi file singoli

            # Se l'utente ha specificato dei layer, usiamo l'intersezione
            if self.filter_layers:
                target_layers = [l for l in self.filter_layers if l in available_layers]
                if len(target_layers) < len(self.filter_layers):
                    print(f"⚠️ Warning: Alcuni layer richiesti non sono nel file .pt. Userò: {target_layers}")
            else:
                target_layers = available_layers

            print(f"🎯 Steering attivo sui layer: {target_layers}")

            # Caricamento Steerers
            if isinstance(data, dict):
                for idx in target_layers:
                    vec = data[idx]
                    vec = vec / (vec.norm() + 1e-8)
                    if vec.dim() == 2 and vec.shape[0] > 1: vec = vec.mean(dim=0, keepdim=True)
                    
                    target_mod = self.mg.model.lm.transformer.layers[idx]
                    steerers.append(DynamicSteering(target_mod, vec))
            else:
                # Caso singolo (compatibilità)
                vec = data
                vec = vec / (vec.norm() + 1e-8)
                if vec.dim() == 2 and vec.shape[0] > 1: vec = vec.mean(dim=0, keepdim=True)
                for idx in target_layers:
                    target_mod = self.mg.model.lm.transformer.layers[idx]
                    steerers.append(DynamicSteering(target_mod, vec))
                    
        except Exception as e: print(f"❌ Error loading vector: {e}"); return

        if not steerers: print("❌ Nessuno steerer attivato."); return

        try:
            df = pd.read_csv(prompts_file, sep=';')
            if max_samples: df = df.head(max_samples)
        except: print("❌ Error reading CSV"); return

        os.makedirs(output_dir, exist_ok=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                pid = str(row.get('ID', row.get('id', i))).strip()
                if 'test_prompt' in row: prompt = str(row['test_prompt']).strip()
                elif 'prompt' in row: prompt = str(row['prompt']).strip()
                else: prompt = str(row.iloc[-1]).strip()
                
                # Nomi file puliti per Evaluation: {ID}_xxx.wav
                base = os.path.join(output_dir, f"{pid}")

                self.mg.generate(prompt, f"{base}_orig")
                
                for s in steerers: s.apply(alpha)
                self.mg.generate(prompt, f"{base}_pos")
                for s in steerers: s.remove()
                
                for s in steerers: s.apply(-alpha)
                self.mg.generate(prompt, f"{base}_neg")
                for s in steerers: s.remove()
                
            except Exception as e:
                print(f"Error {pid}: {e}")
                for s in steerers: s.remove()


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

    def _create_bar_chart(self, x_labels, values, title, y_label, y_limit=(-1.1, 1.1)):
        plt.figure(figsize=(12, 6))
        cmap = plt.get_cmap('bwr')
        norm = mcolors.Normalize(vmin=-1, vmax=1) 
        bar_colors = cmap(norm(values))

        bars = plt.bar(x_labels, values, color=bar_colors, edgecolor='black', width=0.6)
        plt.axhline(0, color='black', linewidth=1.5)
        plt.ylim(y_limit[0], y_limit[1])
        
        plt.title(title, fontsize=16)
        
        # --- MODIFICA RICHIESTA: LABELS ASSI ---
        plt.xlabel("Tracks ID", fontsize=12)  # Fisso
        plt.ylabel(y_label, fontsize=12)      # Parametrico
        # ---------------------------------------

        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)

        y_range = y_limit[1] - y_limit[0]
        offset = y_range * 0.02

        for bar, score in zip(bars, values):
            y_pos = score + offset if score > 0 else score - (offset * 2)
            plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{score:.2f}", 
                     ha='center', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        
        # --- SALVATAGGIO GRAFICO ---
        if self.output_dir:
            # Crea nome file sicuro dal titolo
            safe_name = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
            save_path = os.path.join(self.output_dir, f"plot_{safe_name}.png")
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"📊 Grafico salvato: {save_path}")
        
        plt.show()
        plt.close()

    # --- PLOTTING FUNCTIONS ---
    # Tutte le funzioni ora accettano y_label e la passano avanti
    
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
        # Passiamo y_label
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
        
        # Passiamo y_label
        self._create_bar_chart(labels, deltas, title, y_label, y_limit=(-2.0, 2.0))

    # --- SALVATAGGIO CSV ---
    def save_to_csv(self):
        """Salva il CSV in base alla modalità (TRAIN o FULL)."""
        os.makedirs(self.output_dir, exist_ok=True)
        full_path = os.path.join(self.output_dir, self.csv_filename)

        print(f"\n--- Inizio calcolo e salvataggio CSV su {len(self.ids)} file ---")
        print(f"Modalità TRAIN: {self.train_mode}")
        
        data = []

        for i, audio_id in enumerate(self.ids):
            path_pos = os.path.join(self.audio_folder, f"{audio_id}_pos.wav")
            path_neg = os.path.join(self.audio_folder, f"{audio_id}_neg.wav")
            
            s_pos = self._get_valence_score(path_pos) if os.path.exists(path_pos) else 0.0
            s_neg = self._get_valence_score(path_neg) if os.path.exists(path_neg) else 0.0

            if self.train_mode:
                data.append({
                    "id": audio_id,
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4)
                })
            else:
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

        df = pd.DataFrame(data)
        
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
    def run(cls, audio_folder, output_dir, csv_filename, train_mode=False, num_samples=20, y_label="Score"):
        evaluator = cls(audio_folder, output_dir, csv_filename, train_mode)
        evaluator.evaluate(num_samples_plot=num_samples, y_label=y_label)
        return evaluator

