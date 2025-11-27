from core import MusicGenWrapper
audio = "prova_audio_clap"
mg = MusicGenWrapper(size = 'small', duration=5)
mg.generate("happy guitar solo", audio)


########################CLAP TEST########################

from transformers import pipeline
audio = "prova_audio_clap.wav"
labels = ["happy", "sad", "happy mood", "sad mood"]

audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
output = audio_classifier(audio, candidate_labels=labels)

print("\nRisultati:")
for result in output:
    print(f"Etichetta: {result['label']} | Confidenza: {result['score']:.4f}")
