import torch
import pandas as pd
import os
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import logging
from sklearn.decomposition import PCA
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
# 4. DATASET EXTRACTOR (Versione PCA)
# ==========================================
class DatasetExtractor:
    def __init__(self, model_wrapper, layer_idx=14):
        self.mg = model_wrapper
        self.layer_idx = layer_idx
        self.target_layer = self.mg.model.lm.transformer.layers[layer_idx]
        self.hook = ActivationHook(self.target_layer)
        
    def extract(self, csv_path, save_path, audio_output_dir=None, sep=';', use_pca=True):
        print(f"🏭 Extracting from {csv_path} (PCA={use_pca})...")
        
        try: df = pd.read_csv(csv_path, sep=sep)
        except: print("❌ Error reading CSV"); return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if audio_output_dir: os.makedirs(audio_output_dir, exist_ok=True)

        # Invece di accumulare la somma, salviamo TUTTE le differenze in una lista
        all_differences = []
        count = 0
        self.hook.register()

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                p_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                p_neg = str(row.get('negative_prompt', row.iloc[1])).strip()
                
                # Nomi file audio (opzionali)
                f_pos = os.path.join(audio_output_dir, f"{index}_pos") if audio_output_dir else None
                f_neg = os.path.join(audio_output_dir, f"{index}_neg") if audio_output_dir else None

                # Generazione
                self.mg.generate(p_pos, filename=f_pos)
                vec_pos = self.hook.get_mean_vector()
                self.hook.activations = []

                self.mg.generate(p_neg, filename=f_neg)
                vec_neg = self.hook.get_mean_vector()
                self.hook.activations = []

                if vec_pos is not None and vec_neg is not None:
                    # Calcola differenza grezza
                    diff = vec_pos - vec_neg
                    
                    # Salviamo nella lista per la PCA
                    all_differences.append(diff.cpu()) # Portiamo su CPU per sklearn
                    count += 1
            except Exception as e: print(f"Err {index}: {e}")

        self.hook.remove()

        if len(all_differences) > 0:
            # Stack di tutti i vettori: [N_coppie, 1024]
            X = torch.stack(all_differences).numpy()
            
            if use_pca and len(all_differences) > 1:
                print("🧠 Calcolo PCA per pulire il rumore...")
                # Calcoliamo la componente principale
                pca = PCA(n_components=1)
                pca.fit(X)
                
                # Questo è il vettore pulito [1024]
                final_vector = torch.from_numpy(pca.components_[0]).float()
                
                # CHECK DIREZIONE: La PCA può invertire il segno a caso.
                # Controlliamo se punta nella stessa direzione della media semplice.
                mean_simple = torch.mean(torch.stack(all_differences), dim=0)
                if torch.dot(final_vector, mean_simple) < 0:
                    print("🔄 PCA ha invertito il segno. Correggo...")
                    final_vector = -final_vector
            else:
                # Fallback sulla media semplice se PCA è disattivata o pochi dati
                print("Using simple Mean.")
                final_vector = torch.mean(torch.stack(all_differences), dim=0)

            # Normalizzazione Finale e Shape Fix
            final_vector = final_vector / (final_vector.norm() + 1e-8)
            
            # Assicuriamoci che sia [1, 1024] per l'inferenza
            if final_vector.dim() == 1:
                final_vector = final_vector.unsqueeze(0)

            torch.save(final_vector, save_path)
            print(f"✅ Saved vector: {save_path} | Shape: {final_vector.shape}")
        else:
            print("❌ Nessun vettore estratto.")

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
                
                safe_p = "".join([c for c in prompt if c.isalnum() or c in " _-"])[:20].replace(" ", "_")
                base = os.path.join(output_dir, f"{pid}_{safe_p}")

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