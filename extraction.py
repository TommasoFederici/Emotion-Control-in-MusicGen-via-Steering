import torch
from project_base import MusicGenWrapper

# ==========================================
# 🛠️ CLASSE HOOK (Il "Registratore")
# ==========================================
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

# ==========================================
# 🧪 SCRIPT DI ESTRAZIONE
# ==========================================
def run_extraction():
    print("\n🧪 AVVIO ESTRAZIONE VETTORE EMOZIONE...")
    
    # 1. Carica il modello usando la tua base sicura
    # Generiamo 10 secondi per avere abbastanza dati statistici
    mg = MusicGenWrapper(size='small', duration=10)
    
    # 2. Identifica il Layer 14
    # MusicGen Small ha 24 layer (indice 0-23).
    target_layer_idx = 14
    target_layer = mg.model.lm.transformer.layers[target_layer_idx]
    print(f"📍 Target: Layer {target_layer_idx} (Centro Semantico)")

    # 3. Prepara l'Hook
    hook = ActivationHook(target_layer)
    
    # --- STEP A: ESTRAZIONE POSITIVA (HAPPY) ---
    print("\n🎵 Generazione: CONCETTO HAPPY")
    hook.register() # Inizia a registrare
    
    # Usiamo un prompt molto descrittivo per massimizzare l'effetto
    prompt_happy = "A happy upbeat pop song, joyful melody, major key, energetic, euphoria"
    mg.generate(prompt_happy, "reference_happy")
    
    vec_happy = hook.get_mean_vector()
    print(f"   ✅ Vettore Happy estratto. Shape: {vec_happy.shape}")
    
    hook.remove() # Smetti di registrare per pulire
    
    # --- STEP B: ESTRAZIONE NEGATIVA (SAD) ---
    print("\n🎵 Generazione: CONCETTO SAD")
    hook.register() # Ricomincia a registrare
    
    prompt_sad = "A sad melancholic piano song, depressing atmosphere, minor key, slow tempo, grief"
    mg.generate(prompt_sad, "reference_sad")
    
    vec_sad = hook.get_mean_vector()
    print(f"   ✅ Vettore Sad estratto. Shape: {vec_sad.shape}")
    
    hook.remove() # Pulizia finale

    # --- STEP C: CALCOLO DEL VETTORE DI STEERING ---
    print("\n🧮 Calcolo Matematica dello Steering...")
    
    # Formula: Direzione = (Positivo - Negativo)
    # Questo vettore punta "via dalla tristezza" e "verso la felicità"
    steering_vector = vec_happy - vec_sad
    
    # Normalizzazione (Opzionale ma consigliata)
    # Rende il vettore lungo 1, così possiamo decidere noi l'intensità dopo con un moltiplicatore
    steering_vector = steering_vector / steering_vector.norm()
    print(f"   ✅ Vettore Steering normalizzato. Shape: {steering_vector.shape}")
    
    # Salva il risultato
    filename = "vector_happy_sad_layer14.pt"
    torch.save(steering_vector, filename)
    
    print(f"\n💾 VETTORE SALVATO: {filename}")
    print("   Ora hai la 'chiave' per manipolare le emozioni.")
    print("   Prossimo step: Creare 'inference.py' per usare questo file.")

if __name__ == "__main__":
    run_extraction()