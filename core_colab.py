import torch
import pandas as pd
import os
from tqdm import tqdm
import logging

# Configurazione Base per Colab
logging.getLogger("audiocraft").setLevel(logging.ERROR)

# =========================================================================
# 🎵 SEZIONE 1: MUSICGEN WRAPPER
# =========================================================================
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class MusicGenWrapper:
    def __init__(self, size='small', duration=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 MusicGen ({size}) | Device: {self.device} | Durata: {duration}s")
        # Su Colab non serve nessun parametro strano, xformers viene caricato in automatico
        self.model = MusicGen.get_pretrained(size)
        self.model.set_generation_params(duration=duration)

    def generate(self, prompt, filename=None):
        # Su Colab possiamo tenere i progress bar se vogliamo, ma qui li nascondiamo per pulizia
        wav = self.model.generate([prompt], progress=False)
        
        if filename:
            if filename.endswith(".wav"): filename = filename[:-4]
            path = audio_write(filename, wav[0].cpu(), self.model.sample_rate, strategy="loudness", loudness_headroom_db=16)
        
        return wav

# =========================================================================
# 🔬 SEZIONE 2: ESTRAZIONE (ActivationHook)
# =========================================================================
class ActivationHook:
    def __init__(self, module):
        self.module = module
        self.handle = None
        self.activations = [] 

    def hook_fn(self, module, input, output):
        # Gestione Output Tuple
        if isinstance(output, tuple): output = output[0]
        
        # --- FIX CFG (Classifier Free Guidance) ---
        # Se il batch è > 1 (es. 2: Condizionato + Incondizionato), 
        # prendiamo solo la parte condizionata (Indice 0)
        if output.shape[0] > 1:
            # Prende slice [0:1] per mantenere le dimensioni [1, Time, Dim]
            clean_output = output[0:1] 
        else:
            clean_output = output

        self.activations.append(clean_output.detach().cpu())

    def register(self):
        self.handle = self.module.register_forward_hook(self.hook_fn)
        self.activations = [] 

    def remove(self):
        if self.handle: self.handle.remove()
    
    def get_mean_vector(self):
        if not self.activations: return None
        full = torch.cat(self.activations, dim=1)
        # Media sul tempo
        return full.mean(dim=1).squeeze()

# =========================================================================
# 🎛️ SEZIONE 3: INFERENZA (WeightSteering)
# =========================================================================
class WeightSteering:
    """Modifica i Bias per lo steering. Metodo stabile e matematicamente valido."""
    def __init__(self, module, steering_vector):
        self.module = module
        self.original_bias = None
        
        try:
            device = next(module.parameters()).device
        except:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.steering_vector = steering_vector.to(device).float()
        
    def apply(self, coefficient=1.0):
        # Cerca il target (linear2 o out_proj)
        linear_module = None
        if hasattr(self.module, 'linear2'):
            linear_module = self.module.linear2
        elif hasattr(self.module, 'out_proj'):
            linear_module = self.module.out_proj
        else:
            for m in self.module.modules():
                if isinstance(m, torch.nn.Linear):
                    linear_module = m
        
        if not linear_module: return # Fail silent or raise error

        self.target_module = linear_module 

        # Salva Bias Originale
        if linear_module.bias is not None:
            self.original_bias = linear_module.bias.data.clone()
        else:
            self.original_bias = None
            linear_module.bias = torch.nn.Parameter(
                torch.zeros(linear_module.out_features, device=linear_module.weight.device)
            )
        
        # Applica Delta
        delta = self.steering_vector.view(-1) * coefficient
        
        # Adattamento dimensionale (Safety)
        if delta.shape[0] != linear_module.bias.shape[0]:
            delta = delta[:linear_module.bias.shape[0]]

        linear_module.bias.data = linear_module.bias.data + delta
        
    def remove(self):
        if hasattr(self, 'target_module'):
            if self.original_bias is not None:
                self.target_module.bias.data = self.original_bias
            else:
                self.target_module.bias = None

# =========================================================================
# 🏭 SEZIONE 4: DATASET EXTRACTOR (Automazione)
# =========================================================================
class DatasetExtractor:
    def __init__(self, model_wrapper, layer_idx=14):
        self.mg = model_wrapper
        self.layer_idx = layer_idx
        self.target_layer = self.mg.model.lm.transformer.layers[layer_idx]
        self.hook = ActivationHook(self.target_layer)
        
    def extract(self, csv_path, save_path, sep=';'):
        print(f"\n🏭 ESTRAZIONE DATASET (Layer {self.layer_idx}) -> {save_path}")
        
        try:
            df = pd.read_csv(csv_path, sep=sep)
        except Exception as e:
            print(f"❌ Errore CSV: {e}")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cumulative_vector = None
        count = 0
        self.hook.register()

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Gestione colonne flessibile
                p_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                p_neg = str(row.get('negative_prompt', row.iloc[1])).strip()

                # Generazione silenziosa (non salviamo audio per velocità)
                self.mg.model.generate([p_pos], progress=False)
                vec_pos = self.hook.get_mean_vector()
                self.hook.activations = []

                self.mg.model.generate([p_neg], progress=False)
                vec_neg = self.hook.get_mean_vector()
                self.hook.activations = []

                if vec_pos is not None and vec_neg is not None:
                    diff = vec_pos - vec_neg
                    # Normalizzazione Locale (Cruciale!)
                    diff = diff / (diff.norm() + 1e-8)

                    if cumulative_vector is None: cumulative_vector = diff
                    else: cumulative_vector += diff
                    count += 1
                
            except Exception as e:
                print(f"Errore riga {index}: {e}")
                self.hook.activations = []

        self.hook.remove()

        if cumulative_vector is not None:
            # Media e Normalizzazione Finale
            mean_vector = cumulative_vector / count
            mean_vector = mean_vector / mean_vector.norm()
            torch.save(mean_vector, save_path)
            print(f"✅ Salvato: {save_path} ({count} coppie)")
        else:
            print("❌ Nessun vettore estratto.")

# =========================================================================
# 🚀 SEZIONE 5: DATASET INFERENCE (Automazione)
# =========================================================================
class DatasetInference:
    def __init__(self, model_wrapper, layer_idx=14):
        self.mg = model_wrapper
        self.layer_idx = layer_idx
        self.target_layer = self.mg.model.lm.transformer.layers[layer_idx]

    def run(self, prompts_file, vector_path, output_dir, alpha=1.5, max_samples=None):
        print(f"\n🚀 INFERENZA BATCH (Alpha {alpha})")
        
        # Carica Vettore
        try:
            vec = torch.load(vector_path)
            # Fix Shape [2, 1024] -> [1, 1024]
            if vec.dim() == 2 and vec.shape[0] > 1:
                vec = vec.mean(dim=0, keepdim=True)
            vec = vec / (vec.norm() + 1e-8)
        except Exception as e:
            print(f"❌ Errore vettore: {e}"); return

        # Carica Dataset
        try:
            df = pd.read_csv(prompts_file, sep=';')
            if max_samples: df = df.head(max_samples)
        except: print("❌ Errore CSV prompt"); return

        os.makedirs(output_dir, exist_ok=True)
        steerer = WeightSteering(self.target_layer, vec)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                pid = str(row.get('ID', row.get('id', i))).strip()
                prompt = str(row.get('test_prompt', row.get('prompt', row.iloc[-1]))).strip()
                
                safe_p = "".join([c for c in prompt if c.isalnum() or c in " _-"])[:20].replace(" ", "_")
                base = os.path.join(output_dir, f"{pid}_{safe_p}")

                # Generazione
                self.mg.generate(prompt, f"{base}_orig")
                
                steerer.apply(alpha)
                self.mg.generate(prompt, f"{base}_pos")
                steerer.remove()
                
                steerer.apply(-alpha)
                self.mg.generate(prompt, f"{base}_neg")
                steerer.remove()
                
            except Exception as e:
                print(f"Errore {pid}: {e}")
                steerer.remove()