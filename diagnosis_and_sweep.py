import torch
import os
from core import MusicGenWrapper, WeightSteering

def run_diagnostics():
    print("\n🔍 DIAGNOSTICA VETTORE & SWEEP")
    
    # --- 1. ANALISI DEL VETTORE ---
    try:
        # Cerca il file vettore più recente o specifico
        vector_path = "vector_happy_sad_layer14.pt" 
        if not os.path.exists(vector_path):
            # Fallback su cartella vectors se esiste
            if os.path.exists("vectors"):
                files = os.listdir("vectors")
                if files:
                    vector_path = os.path.join("vectors", files[0])
        
        print(f"📂 Analizzando: {vector_path}")
        vec = torch.load(vector_path)
        
        print(f"   Shape: {vec.shape}")
        print(f"   Min: {vec.min().item():.4f}")
        print(f"   Max: {vec.max().item():.4f}")
        print(f"   Mean: {vec.mean().item():.4f}")
        print(f"   Norma (Lunghezza): {vec.norm().item():.4f}")
        
        if torch.isnan(vec).any():
            print("❌ ALLARME: Il vettore contiene NaN! L'estrazione è fallita.")
            return
        
        # Normalizziamo per sicurezza prima dello sweep
        vec = vec / (vec.norm() + 1e-8)
        print("✅ Vettore normalizzato per lo sweep.")
        
    except Exception as e:
        print(f"❌ Errore caricamento vettore: {e}")
        return

    # --- 2. SWEEP DI INTENSITÀ (Trova il volume giusto) ---
    print("\n🎛️ AVVIO SWEEP INTENSITÀ...")
    mg = MusicGenWrapper(size='small', duration=4) # Breve per fare prima
    
    # Usiamo il Layer 14 come standard
    target_layer = mg.model.lm.transformer.layers[14]
    steerer = WeightSteering(target_layer, vec)
    
    prompt = "A simple acoustic guitar melody"
    os.makedirs("sweep_test", exist_ok=True)
    
    # Proviamo valori MOLTO diversi, anche piccoli
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    print(f"🧪 Generazione varianti per prompt: '{prompt}'")
    
    # Base
    mg.generate(prompt, "sweep_test/0_original")
    
    for alpha in alphas:
        print(f"   👉 Generando Alpha {alpha}...")
        
        # Happy (+)
        mod = steerer.apply(coefficient=alpha)
        mg.generate(prompt, f"sweep_test/happy_alpha_{alpha}")
        steerer.remove()
        
        # Sad (-)
        mod = steerer.apply(coefficient=-alpha)
        mg.generate(prompt, f"sweep_test/sad_alpha_{alpha}")
        steerer.remove()

    print("\n✅ FINE.")
    print("Ascolta la cartella 'sweep_test'.")
    print("Trova il file che ha l'effetto emotivo MA non gracchia.")
    print("Quello sarà il tuo Alpha definitivo.")

if __name__ == "__main__":
    run_diagnostics()