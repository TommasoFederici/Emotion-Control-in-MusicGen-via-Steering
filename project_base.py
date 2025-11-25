import sys
import types
import os
import torch
import torch.nn.functional as F
import warnings
import logging

# ==========================================
# 🔇 BLOCCO SILENZIATORE (PULIZIA)
# ==========================================
# Questo blocco nasconde i FutureWarning e UserWarning inutili di PyTorch/Audiocraft
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XFORMERS_FORCE_DISABLE"] = "1"

# Filtra i warning rossi fastidiosi
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Silenzia i log interni di AudioCraft (info inutili)
logging.getLogger("audiocraft").setLevel(logging.ERROR)

# ==========================================
# 🛡️ BLOCCO WINDOWS BYPASS (XFORMERS)
# ==========================================
if "xformers" not in sys.modules:
    # 1. Creiamo i moduli finti
    mock_xformers = types.ModuleType("xformers")
    mock_xformers.ops = types.ModuleType("xformers.ops")
    
    # 2. Definiamo le funzioni critiche
    def mock_attention(q, k, v, *args, **kwargs):
        kwargs.pop("attn_bias", None)
        kwargs.pop("p", None)
        return F.scaled_dot_product_attention(q, k, v)

    def mock_unbind(input, dim=0):
        return torch.unbind(input, dim)

    class DummyClass:
        def __init__(self, *args, **kwargs): pass

    # 3. Iniettiamo tutto
    mock_xformers.ops.memory_efficient_attention = mock_attention
    mock_xformers.ops.unbind = mock_unbind
    mock_xformers.ops.LowerTriangularMask = DummyClass
    
    sys.modules["xformers"] = mock_xformers
    sys.modules["xformers.ops"] = mock_xformers.ops

# ==========================================
# 🎵 MUSICGEN WRAPPER (CLASSE PRINCIPALE)
# ==========================================
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

# ==========================================
# 🧪 TEST RAPIDO
# ==========================================
if __name__ == "__main__":
    try:
        mg = MusicGenWrapper(duration=3) 
        mg.generate("A clean silent console test", "test_clean")
        print("\n✨ Console pulita! Ora puoi lavorare.")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")