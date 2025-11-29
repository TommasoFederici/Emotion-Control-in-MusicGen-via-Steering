import torch
import os
from VECCHI_FILE.core import MusicGenWrapper, WeightSteering

def run_micro_sweep():
    print("\n🔬 MICRO-SWEEP (Calibrazione di Precisione)")
    
    # 1. Carica Vettore
    try:
        vector_path = "data/vectors/steering_vector_Sad2Happy_layer14_avg50.pt"
        if not os.path.exists(vector_path): vector_path = "vector_happy_sad_layer14.pt"
        
        vec = torch.load(vector_path)
        print(f"📦 Vettore caricato. Shape: {vec.shape}")
        
        # Normalizzazione di sicurezza
        vec = vec / (vec.norm() + 1e-8)
        
    except Exception as e:
        print(f"❌ Errore file: {e}")
        return

    # 2. Carica Modello
    mg = MusicGenWrapper(size='small', duration=5)
    layer_idx = 14
    target_layer = mg.model.lm.transformer.layers[layer_idx]
    
    # --- ANALISI DIMENSIONALE ---
    # Cerca il layer lineare target
    linear_layer = None
    if hasattr(target_layer, 'linear2'): linear_layer = target_layer.linear2
    elif hasattr(target_layer, 'out_proj'): linear_layer = target_layer.out_proj
    else:
        # Cerca il primo lineare disponibile
        for m in target_layer.modules():
            if isinstance(m, torch.nn.Linear):
                linear_layer = m
                break
    
    if linear_layer:
        print(f"\n📊 STATISTICHE LAYER {layer_idx}:")
        
        # FIX ERRORE: Gestione Bias Assente
        if linear_layer.bias is not None:
            bias_mean = linear_layer.bias.data.abs().mean().item()
            print(f"   Valore medio Bias originale: {bias_mean:.6f}")
        else:
            bias_mean = 0.0
            print(f"   ⚠️ Questo layer NON HA BIAS (sarà inizializzato a 0 dallo steering).")
            # Per confronto, guardiamo i pesi
            weight_mean = linear_layer.weight.data.abs().mean().item()
            print(f"   Riferimento (Media Pesi): {weight_mean:.6f}")

        vec_mean = vec.abs().mean().item()
        print(f"   Valore medio Vettore Steering: {vec_mean:.6f}")
        
        if bias_mean > 0:
            ratio = vec_mean / bias_mean
            print(f"   Rapporto Vettore/Bias: {ratio:.1f}x")
        elif 'weight_mean' in locals():
            ratio = vec_mean / weight_mean
            print(f"   Rapporto Vettore/Pesi: {ratio:.1f}x (Indicativo)")

    # 3. SETUP MICRO-SWEEP
    steerer = WeightSteering(target_layer, vec)
    prompt = "A simple acoustic guitar melody"
    output_dir = "micro_sweep_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Testiamo valori DECIMAL (molto piccoli)
    micro_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    
    print(f"\n🧪 Generazione varianti in '{output_dir}'...")
    
    # Base
    mg.generate(prompt, f"{output_dir}/00_original")
    
    for alpha in micro_alphas:
        print(f"   👉 Testing Alpha {alpha}...")
        
        # Happy
        steerer.apply(coefficient=alpha)
        mg.generate(prompt, f"{output_dir}/happy_0.{int(alpha*10)}")
        steerer.remove()
        
        # Sad
        steerer.apply(coefficient=-alpha)
        mg.generate(prompt, f"{output_dir}/sad_0.{int(alpha*10)}")
        steerer.remove()

    print("\n✅ FINE.")
    print("Ascolta i file. Cerca quello che cambia l'emozione SENZA introdurre rumore alla fine.")

if __name__ == "__main__":
    run_micro_sweep()