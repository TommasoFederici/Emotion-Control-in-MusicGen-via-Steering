from core import MusicGenWrapper, DatasetInference

def main():
    # --- CONFIGURAZIONE ---
    prompts_file = "data/Happy_Sad/test_prompt_Happy_Sad.txt"  
    vector_file = "data/vectors/steering_vector_Sad2Happy_layer14_avg50.pt"
    output_dir = "data/Happy_Sad/results_final"
    
    # Parametri
    alpha = 0.3
    duration_sec = 5  # Durata audio generato per ogni prompt
    layer_idx = 14
    

    # --- ESECUZIONE ---
    mg = MusicGenWrapper(size='small', duration=duration_sec)
    inference_engine = DatasetInference(mg, layer_idx=layer_idx)
    
    inference_engine.run(
        prompts_file=prompts_file,
        vector_path=vector_file,
        output_dir=output_dir,
        alpha=alpha
    )

if __name__ == "__main__":
    main()