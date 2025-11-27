import torch
import pandas as pd
import os
from tqdm import tqdm  # Per la barra di caricamento
from VECCHI_FILE.project_base import MusicGenWrapper
from VECCHI_FILE.extraction import ActivationHook

def run_dataset_extraction():
    print("\n🧪 AVVIO ESTRAZIONE MASSIVA (DATASET COMPLETO)...")

    # --- CONFIGURAZIONE ---
    csv_path = "data/Happy_Sad/dataset_prompt_Happy_Sad.csv"
    output_audio_dir = "data/Happy_Sad/train_audio"  # Dove salvare gli audio generati
    vector_output_dir = "data/vectors"               # Dove salvare il vettore finale
    target_layer_idx = 14                            # Il layer scelto
    duration_sec = 5                                 # Durata audio (come nel paper)
    emotion_direction = "Sad2Happy"
    
    # Crea le cartelle se non esistono
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(vector_output_dir, exist_ok=True)

    # 1. Carica il Dataset
    print(f"📂 Lettura dataset da: {csv_path}")
    # Nota: uso sep=';' perché il tuo CSV usa il punto e virgola
    try:
        df = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        print(f"❌ Errore lettura CSV. Controlla il separatore! Errore: {e}")
        return

    print(f"📊 Trovate {len(df)} coppie di prompt.")

    # 2. Carica il Modello
    print("\n🎹 Caricamento MusicGen...")
    mg = MusicGenWrapper(size='small', duration=duration_sec)
    
    # Setup Hook
    target_layer = mg.model.lm.transformer.layers[target_layer_idx]
    hook = ActivationHook(target_layer)
    print(f"📍 Hook collegato al Layer {target_layer_idx}")

    # Variabili per accumulare i vettori
    # Usiamo la somma e poi dividiamo per il numero totale (Media)
    cumulative_steering_vector = None
    count = 0

    print("\n🚀 INIZIO CICLO DI GENERAZIONE E ESTRAZIONE...")
    
    # tqdm crea una barra di progresso bellissima nel terminale
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Pairs"):
        
        pair_id = row['ID']
        # Pulisco le stringhe da eventuali spazi extra
        prompt_happy = str(row['positive_prompt']).strip()
        prompt_sad = str(row['negative_prompt']).strip()

        # Nomi file per salvare l'audio (opzionale ma utile per controllo)
        filename_happy = os.path.join(output_audio_dir, f"pair_{pair_id}_happy")
        filename_sad = os.path.join(output_audio_dir, f"pair_{pair_id}_sad")

        # --- STEP A: HAPPY ---
        hook.register()
        # Generiamo e salviamo l'audio
        mg.generate(prompt_happy, filename_happy)
        vec_happy = hook.get_mean_vector()
        hook.remove()

        # --- STEP B: SAD ---
        hook.register()
        mg.generate(prompt_sad, filename_sad)
        vec_sad = hook.get_mean_vector()
        hook.remove()

        # --- STEP C: DIFFERENZA (Happy - Sad) ---
        current_diff = vec_happy - vec_sad

        # [FIX] Normalizziamo subito per dare a ogni coppia lo stesso "peso"
        # Altrimenti una coppia con valori altissimi comanda su tutte le altre.
        current_diff = current_diff / (current_diff.norm() + 1e-8)

        # --- STEP D: ACCUMULO ---
        if cumulative_steering_vector is None:
            cumulative_steering_vector = current_diff
        else:
            cumulative_steering_vector += current_diff
        
        count += 1
        
        # Pulizia memoria GPU ogni tanto per sicurezza
        if index % 5 == 0:
            torch.cuda.empty_cache()

    # --- FINE CICLO: CALCOLO MEDIA FINALE ---
    print("\n🧮 Calcolo del vettore medio globale...")
    
    # Divide la somma per il numero di coppie per ottenere la MEDIA
    final_steering_vector = cumulative_steering_vector / count

    # Normalizzazione (Rende la lunghezza del vettore = 1)
    # Fondamentale per applicare poi un coefficiente 'alpha' preciso
    final_steering_vector = final_steering_vector / final_steering_vector.norm()

    # Salvataggio
    save_path = os.path.join(vector_output_dir, f"steering_vector_{emotion_direction}_layer{target_layer_idx}_avg{count}.pt")
    torch.save(final_steering_vector, save_path)

    print(f"\n✅ COMPLETATO!")
    print(f"💾 Vettore salvato in: {save_path}")
    print(f"🎧 Audio salvati in: {output_audio_dir}")
    print(f"📈 Coppie processate: {count}")
    print(f"Shape vettore finale: {final_steering_vector.shape}")

if __name__ == "__main__":
    run_dataset_extraction()