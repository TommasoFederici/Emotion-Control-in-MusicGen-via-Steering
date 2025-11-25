import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# 1. Carica il modello pre-addestrato
# 'small' (300M), 'medium' (1.5B), 'large' (3.3B)
print("Caricamento modello...")
model = MusicGen.get_pretrained('small')

# Imposta i parametri di generazione
model.set_generation_params(duration=5)  # Generiamo solo 5 secondi per test rapido

# 2. Definisci i prompt (Simuliamo lo Studente A)
descriptions = [
    "A happy upbeat pop song", 
    "A sad melancholic piano melody"
]

# 3. Generazione
print("Generazione in corso...")
wav = model.generate(descriptions)  # Restituisce un tensore (batch, canali, tempo)

# 4. Salvataggio su disco
for idx, one_wav in enumerate(wav):
    # Salviamo a 32kHz (standard di MusicGen)
    audio_write(f'test_output_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")

print("Fatto! Controlla i file .wav nella cartella.")