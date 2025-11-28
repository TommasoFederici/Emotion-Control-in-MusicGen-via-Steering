import os
from core import MusicGenWrapper
from transformers import pipeline

# Configurazione cartella di output
output_folder = os.path.join("data", "Happy_Sad", "audio")

# Crea la cartella se non esiste (crea anche le cartelle padre se necessario)
os.makedirs(output_folder, exist_ok=True)

# Inizializzazione wrapper
mg = MusicGenWrapper(size='small', duration=3)

tracce_totali = 60

# Ciclo da 1 a 20 (20 * 3 = 60 tracce totali)
for n in range(1, 21):
    
    # Costruiamo i percorsi completi per i file
    path_origin = os.path.join(output_folder, f"{n}_orig")
    path_pos = os.path.join(output_folder, f"{n}_pos")
    path_neg = os.path.join(output_folder, f"{n}_neg")
    
    # 1. Anchor (Neutro)
    print(f"Generando: {path_origin}...")
    mg.generate("guitar solo", path_origin)

    # 2. Positive (Felice)
    print(f"Generando: {path_pos}...")
    mg.generate("happy guitar solo", path_pos)

    # 3. Negative (Triste)
    print(f"Generando: {path_neg}...")
    mg.generate("sad guitar solo", path_neg)

print(f"Generazione completata: {tracce_totali} tracce salvate in '{output_folder}'.")