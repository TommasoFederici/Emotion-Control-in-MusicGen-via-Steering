import torch
import os
from core import MusicGenWrapper # Usiamo la base sicura senza toccarla

# ==========================================
# 🩹 PATCH LOCALE (Anti-Crash per Windows)
# ==========================================
# Ri-applichiamo la patch per essere sicuri al 100% che questo script non crashi
# quando usiamo gli hook dinamici.
from audiocraft.modules import transformer
def patched_get_mask(self, length, device, dtype):
    return torch.zeros((length, length), device=device, dtype=dtype)
transformer.StreamingMultiheadAttention._get_mask = patched_get_mask
print("🛡️ Patch dinamica attivata per questo test.")

# ==========================================
# 🧪 NUOVA CLASSE: DYNAMIC STEERING (Paper Style)
# ==========================================
class DynamicSteeringHook:
    """
    Implementazione fedele al paper: Activation Steering sul Residual Stream.
    Usa la formula di normalizzazione per evitare distorsioni.
    """
    def __init__(self, module, steering_vector, alpha=1.0):
        self.module = module
        self.handle = None
        
        # Gestione device
        try:
            device = next(module.parameters()).device
        except:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Il vettore deve essere [1, 1, Dim] per il broadcasting corretto su [Batch, Time, Dim]
        self.vector = steering_vector.to(device).float()
        if self.vector.dim() == 2:
            self.vector = self.vector.unsqueeze(1) # Diventa [1, 1, 1024]
            
        self.alpha = alpha

    def hook_fn(self, module, input, output):
        # MusicGen Output: (Tensor, Cache)
        if isinstance(output, tuple):
            h_old = output[0] # Il "Residual Stream" originale
            other = output[1:]
        else:
            h_old = output
            other = ()

        # --- FORMULA DEL PAPER ---
        # h_new = (h_old + alpha * v) / (1 + alpha)
        # Questa divisione impedisce che i valori crescano troppo (distorsione)
        
        h_new = (h_old + (self.alpha * self.vector)) / (1 + self.alpha)
        
        # Ricostruiamo l'output
        if other:
            return (h_new,) + other
        return h_new

    def register(self):
        self.handle = self.module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()

# ==========================================
# 🎧 TEST DI INFERENZA
# ==========================================
def run_dynamic_test():
    print("\n🎧 AVVIO TEST DINAMICO (Paper Implementation)...")

    # 1. Carica Vettore
    vector_path = "data/vectors/steering_vector_Sad2Happy_layer14_avg50.pt"
    # Fallback path
    if not os.path.exists(vector_path): 
        vector_path = "vector_happy_sad_layer14.pt"
        
    try:
        vec = torch.load(vector_path)
        print(f"📦 Vettore caricato: {vector_path}")
        
        # Fix shape se necessario (per sicurezza)
        if vec.dim() == 2 and vec.shape[0] > 1:
             vec = vec.mean(dim=0, keepdim=True)
        # Normalizzazione
        vec = vec / (vec.norm() + 1e-8)
        
    except Exception as e:
        print(f"❌ Errore vettore: {e}")
        return

    # 2. Carica Modello
    mg = MusicGenWrapper(size='small', duration=5)
    target_layer = mg.model.lm.transformer.layers[14] # Layer 14
    
    prompt = "A simple acoustic guitar melody"
    print(f"🎹 Prompt: {prompt}")

    # --- TEST VARI ALPHA ---
    # Col metodo del paper, possiamo osare alpha più alti perché dividiamo per (1+alpha)
    alphas = [1.0, 3.0, 5.0, 10.0] 
    
    os.makedirs("dynamic_test_results", exist_ok=True)
    
    # Base
    mg.generate(prompt, "dynamic_test_results/00_base")

    for a in alphas:
        print(f"\n👉 Testing Alpha {a}...")
        
        # HAPPY (+ Alpha)
        hook = DynamicSteeringHook(target_layer, vec, alpha=a)
        hook.register()
        mg.generate(prompt, f"dynamic_test_results/happy_a{a}")
        hook.remove()
        
        # SAD (Usiamo un alpha negativo o giriamo il vettore?)
        # Il paper dice: aggiungere il vettore. Per Sad usiamo il vettore opposto (-vec).
        # Nota: La formula diviso (1+alpha) funziona meglio con alpha positivi.
        # Quindi passiamo il vettore invertito (-vec) con alpha positivo.
        hook_sad = DynamicSteeringHook(target_layer, -vec, alpha=a)
        hook_sad.register()
        mg.generate(prompt, f"dynamic_test_results/sad_a{a}")
        hook_sad.remove()

    print("\n✅ FINE TEST.")
    print("Ascolta 'dynamic_test_results'.")
    print("Con questa formula, l'audio NON dovrebbe distorcere nemmeno a alpha=5.0.")

if __name__ == "__main__":
    run_dynamic_test()