from core import MusicGenWrapper, DatasetExtractor

def main():
    # 1. Configurazione
    csv_file = "data/Happy_Sad/dataset_prompt_Happy_Sad.csv"
    output_vec = "data/vectors/steering_vector_Sad2Happy_layer14_avg50.pt"
    # Opzionale: metti None se non vuoi salvare 100 file audio e riempire il disco
    #audio_dir = "data/Happy_Sad/train_audio"
    audio_dir = None

    duration_sec = 5        # Durata audio per ogni generazione 
    layer_target_idx = 14   # Layer da cui estrarre il vettore
    
    # 2. Estrazione
    mg = MusicGenWrapper(size='small', duration=duration_sec)
    extractor = DatasetExtractor(mg, layer_idx=layer_target_idx)
    
    extractor.extract(
        csv_path=csv_file, 
        save_path=output_vec, 
        audio_output_dir=audio_dir # Passa None per andare più veloce
    )

if __name__ == "__main__":
    main()