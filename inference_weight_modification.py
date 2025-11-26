import torch
from project_base import MusicGenWrapper
import copy

# ==========================================
# 🎯 WEIGHT STEERING (Modifica Bias Permanente)
# ==========================================
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

# ==========================================
# 🎧 ESPERIMENTO CON WEIGHT MODIFICATION
# ==========================================
def run_inference():
    print("\n🎧 AVVIO TEST DI STEERING (Weight Modification)...")
    print("🔧 Strategia: Modifica temporanea del bias del layer")

    # 1. Carica vettore
    try:
        vector_path = "vectors/steering_vector_Sad2Happy_layer14_avg50.pt" 
        steering_vector = torch.load(vector_path)
        print(f"📦 Vettore caricato: {vector_path}")
    except FileNotFoundError:
        try:
            vector_path = "emotion_vector.pt"
            steering_vector = torch.load(vector_path)
            print(f"📦 Vettore caricato: {vector_path}")
        except FileNotFoundError:
            print(f"❌ ERRORE: Non trovo il file .pt!")
            return

    # 2. Carica modello
    mg = MusicGenWrapper(size='small', duration=5)
    
    # 3. Target layer
    target_layer_idx = 14
    target_layer = mg.model.lm.transformer.layers[target_layer_idx]
    print(f"✅ Target: Layer {target_layer_idx}")

    # 4. Prepara steering
    steerer = WeightSteering(target_layer, steering_vector)

    # 5. Prompt
    neutral_prompt = "A simple acoustic guitar melody, recording studio quality"
    print(f"\n🎹 Prompt Base: '{neutral_prompt}'")

    # --- TEST A: BASE ---
    print("\n1️⃣  Generazione BASE (Originale)...")
    mg.generate(neutral_prompt, "output_original")

    # --- TEST B: HAPPY ---
    print("\n2️⃣  Generazione HAPPY (Steering +2.0)...")
    
    linear_mod = steerer.apply_steering(coefficient=2.0)
    try:
        mg.generate(neutral_prompt, "output_steered_happy")
        print("   ✅ SUCCESSO! Weight steering funziona!")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    finally:
        steerer.remove_steering(linear_mod)

    # --- TEST C: SAD ---
    print("\n3️⃣  Generazione SAD (Steering -2.0)...")
    
    linear_mod = steerer.apply_steering(coefficient=-2.0)
    try:
        mg.generate(neutral_prompt, "output_steered_sad")
        print("   ✅ SUCCESSO!")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    finally:
        steerer.remove_steering(linear_mod)

    print("\n" + "="*60)
    print("✅ TEST COMPLETATO")
    print("📂 Controlla i file output_*.wav")
    print("="*60)
    print("\n💡 Se funziona, puoi sperimentare con:")
    print("   • Coefficienti più alti: 5.0, 10.0")
    print("   • Layer diversi: 12, 16, 18")
    print("   • Prompts più complessi")

if __name__ == "__main__":
    run_inference()