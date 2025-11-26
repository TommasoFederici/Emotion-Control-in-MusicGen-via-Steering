import sys
import types
import os
import torch
import torch.nn.functional as F
import warnings
import logging

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
    """Gestisce il caricamento e la generazione del modello."""
    def __init__(self, size='small', duration=5):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            self.device = "cpu"
            
        print(f"🚀 MusicGen ({size}) | Device: {self.device} | Durata: {duration}s")
        
        self.model = MusicGen.get_pretrained(size)
        self.model.set_generation_params(duration=duration)

    def generate(self, prompt, filename=None):
        print(f"🎹 Generando: '{prompt}'...")
        wav = self.model.generate([prompt])
        
        if filename:
            if filename.endswith(".wav"): filename = filename[:-4]
            path = audio_write(filename, wav[0].cpu(), self.model.sample_rate, strategy="loudness", loudness_headroom_db=16)
            print(f"💾 Salvato: {path}")
        
        return wav


# =========================================================================
# 🔬 SEZIONE 3: ESTRAZIONE (ActivationHook)
# =========================================================================
class ActivationHook:
    """Legge le attivazioni interne per calcolare il vettore."""
    def __init__(self, module):
        self.module = module
        self.handle = None
        self.activations = [] 

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple): output = output[0]
        self.activations.append(output.detach().cpu()) # Clone non necessario se detachiamo subito

    def register(self):
        self.handle = self.module.register_forward_hook(self.hook_fn)
        self.activations = [] 

    def remove(self):
        if self.handle: self.handle.remove()
    
    def get_mean_vector(self):
        if not self.activations: return None
        full = torch.cat(self.activations, dim=1)
        return full.mean(dim=1).squeeze()


# =========================================================================
# 🎛️ SEZIONE 4: INFERENZA (WeightSteering)
# =========================================================================
class WeightSteering:
    """Modifica i pesi (Bias) per applicare lo steering in modo stabile."""
    def __init__(self, module, steering_vector):
        self.module = module
        self.original_bias = None
        
        # Trova device in modo sicuro
        try:
            device = next(module.parameters()).device
        except:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.steering_vector = steering_vector.to(device).float()
        
    def apply(self, coefficient=1.0):
        """Applica lo steering sommando il vettore al bias."""
        linear_module = None
        
        # Logica di ricerca del layer lineare corretto
        if hasattr(self.module, 'linear2'):
            linear_module = self.module.linear2
        elif hasattr(self.module, 'out_proj'):
            linear_module = self.module.out_proj
        else:
            for m in self.module.modules():
                if isinstance(m, torch.nn.Linear):
                    linear_module = m
        
        if not linear_module:
            raise ValueError("Nessun layer lineare trovato!")

        self.target_module = linear_module 

        # Salva stato originale
        if linear_module.bias is not None:
            self.original_bias = linear_module.bias.data.clone()
        else:
            self.original_bias = None
            linear_module.bias = torch.nn.Parameter(
                torch.zeros(linear_module.out_features, device=linear_module.weight.device)
            )
        
        # Calcola delta
        delta = self.steering_vector.view(-1) * coefficient
        
        # Gestione dimensionale
        if delta.shape[0] != linear_module.bias.shape[0]:
            delta = delta[:linear_module.bias.shape[0]]

        # Applica
        linear_module.bias.data = linear_module.bias.data + delta
        
    def remove(self):
        """Ripristina stato originale."""
        if hasattr(self, 'target_module'):
            if self.original_bias is not None:
                self.target_module.bias.data = self.original_bias
            else:
                self.target_module.bias = None