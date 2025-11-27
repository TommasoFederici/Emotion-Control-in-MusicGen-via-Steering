from core import MusicGenWrapper
audio = "prova_audio_clap"
mg = MusicGenWrapper(size = 'small', duration=5)
mg.generate("happy guitar solo", audio)

audio = "prova_audio_clap.wav"
########################CLAP TEST########################
from datasets import load_dataset
from transformers import pipeline

'''# 1. Carica un dataset di esempio (ESC50 contiene suoni ambientali)
# Nota: La prima volta ci metterà un po' a scaricarlo
print("Caricamento dataset...")
dataset = load_dataset("ashraq/esc50", split="train")

# 2. Selezioniamo un file audio specifico (l'ultimo del dataset)
audio = dataset[-1]["audio"]["array"]'''


# 3. Inizializziamo la pipeline CLAP
# Questo scaricherà automaticamente il modello "laion/clap-htsat-unfused"
print("Caricamento modello CLAP...")
audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")


# 4. Definiamo le etichette tra cui il modello deve scegliere
# CLAP è "Zero-Shot", quindi puoi inventare qualsiasi etichetta in inglese!
labels = ["happy", "sad", "happy mood", "sad mood"]

# 5. Eseguiamo la classificazione
output = audio_classifier(audio, candidate_labels=labels)

# 6. Stampiamo il risultato
print("\nRisultati:")
for result in output:
    print(f"Etichetta: {result['label']} | Confidenza: {result['score']:.4f}")
