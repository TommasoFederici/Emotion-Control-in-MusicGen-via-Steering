import sys
import types
import os

# --- INIZIO BYPASS TOTALI ---
os.environ["XFORMERS_FORCE_DISABLE"] = "1"

# Se xformers non c'è, lo creiamo "a mano" per ingannare AudioCraft
if "xformers" not in sys.modules:
    mock_xformers = types.ModuleType("xformers")
    mock_xformers.ops = types.ModuleType("xformers.ops")
    
    # 1. Funzione di attenzione finta (usa quella di PyTorch)
    def mock_attention(q, k, v, *args, **kwargs):
        import torch.nn.functional as F
        # Rimuoviamo argomenti che PyTorch non accetta
        kwargs.pop("attn_bias", None) 
        return F.scaled_dot_product_attention(q, k, v)

    # 2. Maschera triangolare finta (per evitare l'errore LowerTriangularMask)
    class MockLowerTriangularMask:
        def __init__(self, *args, **kwargs):
            pass

    # Iniettiamo le funzioni nel modulo finto
    mock_xformers.ops.memory_efficient_attention = mock_attention
    mock_xformers.ops.LowerTriangularMask = MockLowerTriangularMask
    
    # Registriamo il modulo nel sistema
    sys.modules["xformers"] = mock_xformers
    sys.modules["xformers.ops"] = mock_xformers.ops

print("✅ Bypass xformers AVANZATO attivo.")

# --- FINE BYPASS ---

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test Generazione
try:
    print("\n1. Caricamento Modello...")
    model = MusicGen.get_pretrained('small')
    
    print("2. Generazione...")
    model.set_generation_params(duration=2)
    wav = model.generate(["A hip hop beat"])
    
    print("3. Salvataggio...")
    audio_write('test_bypass', wav[0].cpu(), model.sample_rate, strategy="loudness")
    print("✅ SUCCESS! Audio generato.")
    
except Exception as e:
    print(f"\n❌ ERRORE: {e}")