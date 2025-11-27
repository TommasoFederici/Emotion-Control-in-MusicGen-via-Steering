import sys
import types
import os
import torch
import torch.nn.functional as F
import warnings
import logging
import pandas as pd
from tqdm import tqdm


# =========================================================================
# 🛡️ SEZIONE 1: SETUP AMBIENTE (Windows & Audio Fix)
# =========================================================================
def setup_environment():
    # 1. Silenzia Warning e Log inutili
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["XFORMERS_FORCE_DISABLE"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.getLogger("audiocraft").setLevel(logging.ERROR)

    # 2. Bypass Xformers su Windows
    if "xformers" not in sys.modules:
        mock_xformers = types.ModuleType("xformers")
        mock_xformers.ops = types.ModuleType("xformers.ops")
        
        # --- FUNZIONE CRITICA PER AUDIO PULITO ---
        # Questa funzione intercetta la chiamata e rimuove la maschera sporca
        def mock_attention(q, k, v, *args, **kwargs):
            kwargs.pop("attn_bias", None)
            kwargs.pop("p", None)
            return F.scaled_dot_product_attention(q, k, v)

        def mock_unbind(input, dim=0):
            return torch.unbind(input, dim)

        class DummyClass:
            def __init__(self, *args, **kwargs): pass

        mock_xformers.ops.memory_efficient_attention = mock_attention
        mock_xformers.ops.unbind = mock_unbind
        mock_xformers.ops.LowerTriangularMask = DummyClass
        
        sys.modules["xformers"] = mock_xformers
        sys.modules["xformers.ops"] = mock_xformers.ops

# Eseguiamo il setup subito
setup_environment()


# =========================================================================
# 🎵 SEZIONE 2: MUSICGEN WRAPPER
# =========================================================================
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class MusicGenWrapper:
    def __init__(self, size='small', duration=5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Print pulito, solo info essenziali
        print(f"🚀 MusicGen ({size}) | Device: {self.device} | Durata: {duration}s")
        
        self.model = MusicGen.get_pretrained(size)
        self.model.set_generation_params(duration=duration)

    def generate(self, prompt, filename="output"):
        print(f"🎹 Generando: '{prompt}'...")
        wav = self.model.generate([prompt])
        
        # Salvataggio
        path = audio_write(filename, wav[0].cpu(), self.model.sample_rate, strategy="loudness", loudness_headroom_db=16)
        print(f"💾 Salvato: {filename}.wav")
        return wav



# =========================================================================
# 🔬 SEZIONE 3: ESTRAZIONE (ActivationHook)
# =========================================================================
class ActivationHook:
    """
    Si aggancia a un layer e salva tutto quello che ci passa attraverso.
    """
    def __init__(self, module):
        self.module = module
        self.handle = None
        self.activations = [] 

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple): output = output[0]
        # --- FIX CFG: PRENDIAMO SOLO IL PRIMO ELEMENTO ---
        # MusicGen usa Classifier Free Guidance (CFG).
        # Output shape: [Batch=2, Time, Dim]
        # Batch 0 = Condizionato (Testo) -> QUELLO CHE VOGLIAMO
        # Batch 1 = Incondizionato (Null) -> SPORCIZIA DA SCARTARE
        
        # Se il batch è maggiore di 1, prendiamo solo il primo elemento
        if output.shape[0] > 1:
            clean_output = output[0:1] # Diventa [1, Time, Dim]
        else:
            clean_output = output

        self.activations.append(clean_output.detach().cpu())

    def register(self):
        # Attacca l'hook al layer
        self.handle = self.module.register_forward_hook(self.hook_fn)
        self.activations = [] # Resetta il buffer

    def remove(self):
        # Stacca l'hook (pulizia fondamentale)
        if self.handle:
            self.handle.remove()
            self.handle = None
    
    def get_mean_vector(self):
        """
        Concatena tutti i passaggi temporali e calcola la media.
        Ritorna un vettore di dimensione [1, Hidden_Dim] (es. 1024).
        """
        if not self.activations: 
            return None
        
        # 1. Concatena la lista di tensori lungo la dimensione temporale
        # Shape attesa: [Batch, Total_Time_Steps, 1024]
        full_tensor = torch.cat(self.activations, dim=1) 
        
        # 2. Fai la media su tutto il tempo (dim=1)
        # Otteniamo un singolo vettore che rappresenta "l'essenza media" del brano
        mean_vector = full_tensor.mean(dim=1)
        
        return mean_vector


# =========================================================================
# 🎛️ SEZIONE 4: INFERENZA (WeightSteering)
# =========================================================================
class WeightSteering:
    """
    Invece di usare hook durante inference, modifichiamo temporaneamente
    il BIAS del layer per "spostare" tutte le attivazioni.
    
    Questo è matematicamente equivalente ma evita problemi di cache.
    """
    def __init__(self, module, steering_vector):
        self.module = module
        self.original_bias = None
        self.steering_vector = steering_vector.to(next(module.parameters()).device).float()
        
    def apply_steering(self, coefficient=1.0):
        """
        Aggiunge il vettore di steering al bias del layer.
        Se il layer non ha bias, lo creiamo.
        """
        # Trova il primo linear layer nel modulo
        linear_module = None
        if hasattr(self.module, 'linear2'):
            linear_module = self.module.linear2
        elif hasattr(self.module, 'linear1'):
            linear_module = self.module.linear1
        elif isinstance(self.module, torch.nn.Linear):
            linear_module = self.module
        else:
            # Cerca ricorsivamente
            for submodule in self.module.modules():
                if isinstance(submodule, torch.nn.Linear):
                    linear_module = submodule
                    break
        
        if linear_module is None:
            raise ValueError("Nessun layer Linear trovato nel modulo")
        
        # Salva bias originale
        if linear_module.bias is not None:
            self.original_bias = linear_module.bias.data.clone()
        else:
            # Crea bias zero se non esiste
            self.original_bias = None
            linear_module.bias = torch.nn.Parameter(
                torch.zeros(linear_module.out_features, device=linear_module.weight.device)
            )
        
        # Applica steering al bias
        # Il vettore ha shape [1, hidden_dim], il bias ha shape [hidden_dim]
        steering_flat = self.steering_vector.squeeze() * coefficient
        linear_module.bias.data = linear_module.bias.data + steering_flat
        
        return linear_module
    
    def remove_steering(self, linear_module):
        """Ripristina il bias originale"""
        if self.original_bias is not None:
            linear_module.bias.data = self.original_bias
        else:
            # Se avevamo creato il bias, lo rimuoviamo
            linear_module.bias = None


# =========================================================================
# 🏭 SEZIONE 5: DATASET EXTRACTOR (CLASSE)
# =========================================================================
class DatasetExtractor:
    """
    Gestisce l'estrazione di vettori di steering da un dataset CSV.
    Include la logica di normalizzazione locale per evitare rumore.
    """
    def __init__(self, model_wrapper, layer_idx=14):
        """
        Args:
            model_wrapper: Istanza di MusicGenWrapper già caricata.
            layer_idx: Indice del layer da cui estrarre (default 14).
        """
        self.mg = model_wrapper
        self.layer_idx = layer_idx
        self.target_layer = self.mg.model.lm.transformer.layers[layer_idx]
        self.hook = ActivationHook(self.target_layer)
        
    def extract(self, csv_path, save_path, audio_output_dir=None, sep=';'):
        """
        Esegue l'estrazione su tutto il CSV.
        
        Args:
            csv_path: Percorso al file CSV (colonne: positive_prompt, negative_prompt).
            save_path: Dove salvare il file .pt finale.
            audio_output_dir: (Opzionale) Se specificato, salva gli audio generati per debug.
            sep: Separatore del CSV (default ';').
        """
        print(f"\n🏭 AVVIO ESTRAZIONE DATASET (Layer {self.layer_idx})")
        print(f"📂 Input: {csv_path}")
        
        # 1. Caricamento Dataset
        try:
            df = pd.read_csv(csv_path, sep=sep)
            print(f"📊 Trovate {len(df)} coppie.")
        except Exception as e:
            print(f"❌ Errore lettura CSV: {e}")
            return

        # Setup cartelle
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if audio_output_dir:
            os.makedirs(audio_output_dir, exist_ok=True)

        # 2. Ciclo di Estrazione
        cumulative_vector = None
        count = 0
        self.hook.register()

        try:
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Estrazione"):
                # Parsing prompt
                prompt_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                prompt_neg = str(row.get('negative_prompt', row.iloc[1])).strip()
                
                # Nomi file audio (se richiesti)
                file_pos = None
                file_neg = None
                if audio_output_dir:
                    pair_id = row.get('ID', index)
                    file_pos = os.path.join(audio_output_dir, f"pair_{pair_id}_pos")
                    file_neg = os.path.join(audio_output_dir, f"pair_{pair_id}_neg")

                # --- A. Generazione POSITIVA ---
                self.mg.generate(prompt_pos, filename=file_pos) # Salva solo se file_pos non è None
                vec_pos = self.hook.get_mean_vector()
                self.hook.activations = [] # Reset buffer

                # --- B. Generazione NEGATIVA ---
                self.mg.generate(prompt_neg, filename=file_neg)
                vec_neg = self.hook.get_mean_vector()
                self.hook.activations = [] # Reset buffer

                # --- C. Calcolo Differenza & Normalizzazione Locale ---
                if vec_pos is not None and vec_neg is not None:
                    diff = vec_pos - vec_neg
                    
                    # FIX ROBUSTEZZA: Normalizziamo ogni coppia singolarmente
                    # Così una canzone "rumorosa" non pesa più delle altre.
                    diff_norm = diff / (diff.norm() + 1e-8)

                    if cumulative_vector is None:
                        cumulative_vector = diff_norm
                    else:
                        cumulative_vector += diff_norm
                    
                    count += 1
                
                # Pulizia VRAM ogni 5 step
                if index % 5 == 0:
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\n⚠️ Interrotto dall'utente! Salvo il parziale...")
        
        finally:
            self.hook.remove()

        # 3. Media e Salvataggio Finale
        if cumulative_vector is not None and count > 0:
            mean_vector = cumulative_vector / count
            mean_vector = mean_vector / mean_vector.norm() # Normalizzazione finale
            
            torch.save(mean_vector, save_path)
            print(f"\n✅ VETTORE SALVATO: {save_path}")
            print(f"📈 Basato su {count} coppie valide.")
            print(f"📐 Shape: {mean_vector.shape}")
        else:
            print("❌ Nessun dato estratto.")


# =========================================================================
# 🚀 SEZIONE 6: DATASET INFERENCE (CLASSE)
# =========================================================================
class DatasetInference:
    """
    Applica lo steering su una lista di prompt di test (Batch Inference).
    Genera 3 varianti per ogni prompt: Originale, Positivo, Negativo.
    """
    def __init__(self, model_wrapper, layer_idx=14):
        self.mg = model_wrapper
        self.layer_idx = layer_idx
        self.target_layer = self.mg.model.lm.transformer.layers[layer_idx]

    def run(self, prompts_file, vector_path, output_dir, alpha=1.5):
        """
        Esegue il test su tutti i prompt nel file txt.
        
        Args:
            prompts_file: Percorso al file .txt (un prompt per riga).
            vector_path: Percorso al file .pt del vettore.
            output_dir: Cartella dove salvare i risultati.
            alpha: Intensità dello steering (es. 1.5).
        """
        print(f"\n🚀 AVVIO INFERENZA BATCH (Layer {self.layer_idx}, Alpha {alpha})")
        
        # 1. Caricamento e Validazione Vettore
        try:
            steering_vector = torch.load(vector_path)
            print(f"📦 Vettore caricato: {vector_path} | Shape: {steering_vector.shape}")
            
            # FIX SHAPE ROBUSTO: Se il vettore è [2, 1024], prendiamo la media o il primo
            if steering_vector.dim() == 2 and steering_vector.shape[0] > 1:
                print("⚠️ Vettore con batch > 1 rilevato. Appiattimento...")
                steering_vector = steering_vector.mean(dim=0, keepdim=True)
            
            # Normalizzazione di sicurezza al momento dell'uso
            steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
            
        except Exception as e:
            print(f"❌ Errore caricamento vettore: {e}")
            return

        # 2. Preparazione Prompt
        if not os.path.exists(prompts_file):
            print(f"❌ Errore: Manca il file prompt {prompts_file}")
            return
            
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
            
        print(f"📋 Prompt da processare: {len(prompts)}")
        os.makedirs(output_dir, exist_ok=True)

        # 3. Inizializzazione Steerer
        steerer = WeightSteering(self.target_layer, steering_vector)

        # 4. Ciclo di Generazione
        for i, prompt in enumerate(tqdm(prompts, desc="Batch Inference")):
            # Nome file pulito
            safe_prompt = "".join([c for c in prompt if c.isalnum() or c in " _-"])[:30].replace(" ", "_")
            base_name = os.path.join(output_dir, f"{i:02d}_{safe_prompt}")

            try:
                # A. Originale (Base)
                # Utile per avere un confronto "ground truth"
                self.mg.generate(prompt, f"{base_name}_original")

                # B. Positivo (Happy)
                steerer.apply(coefficient=alpha)
                self.mg.generate(prompt, f"{base_name}_pos")
                steerer.remove() # Reset immediato

                # C. Negativo (Sad)
                steerer.apply(coefficient=-alpha)
                self.mg.generate(prompt, f"{base_name}_neg")
                steerer.remove() # Reset immediato
                
            except Exception as e:
                print(f"❌ Errore sul prompt '{prompt}': {e}")
                # Assicuriamoci di rimuovere lo steering in caso di errore per non inquinare il prossimo
                steerer.remove() 

        print(f"\n✅ INFERENZA COMPLETATA.")
        print(f"📂 Risultati in: {output_dir}")