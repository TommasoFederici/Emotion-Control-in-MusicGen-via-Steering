import numpy as np
from scipy.io import wavfile
import os
import random

# --- CONFIGURAZIONE GLOBALE ---
BASE_DIR = "data/melodies"
EXT_DIR = os.path.join(BASE_DIR, "extraction") # Per il training
TEST_DIR = os.path.join(BASE_DIR, "test")      # Per l'inferenza
SAMPLE_RATE = 32000

# Creazione cartelle
os.makedirs(EXT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# --- UTILS DI GENERAZIONE ---
def generate_tone(freq, duration, sr=32000, waveform='sine'):
    """Genera un singolo tono con envelope ADSR."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if waveform == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif waveform == 'triangle':
        # Onda triangolare (suono un po' più ricco ma neutro)
        wave = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    
    # Envelope (Attacco morbido per evitare click)
    envelope = np.concatenate([
        np.linspace(0, 1, int(sr * 0.05)),       # Attack
        np.ones(int(sr * (duration - 0.1))),     # Sustain
        np.linspace(1, 0, int(sr * 0.05))        # Release
    ])
    
    # Safety check dimensioni
    min_len = min(len(wave), len(envelope))
    return (wave[:min_len] * envelope[:min_len]).astype(np.float32)

def save_sequence(freqs, durations, folder, name, target_len_sec=10):
    """Concatena note, loopa fino a target_len e salva."""
    audio = []
    for f, d in zip(freqs, durations):
        audio.append(generate_tone(f, d))
    
    full_audio = np.concatenate(audio)
    
    # Loop per riempire la durata richiesta
    target_samples = int(SAMPLE_RATE * target_len_sec)
    if len(full_audio) < target_samples:
        repeats = target_samples // len(full_audio) + 1
        full_audio = np.tile(full_audio, repeats)
    
    full_audio = full_audio[:target_samples]
    
    # Normalizzazione (-1 a 1)
    full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-9)
    
    path = os.path.join(folder, f"{name}.wav")
    wavfile.write(path, SAMPLE_RATE, full_audio)
    print(f"🎹 Salvato: {path}")
    return path

# --- FREQUENZE BASE (Scala C4) ---
C4=261.63; D4=293.66; E4=329.63; F4=349.23
G4=392.00; A4=440.00; B4=493.88; C5=523.25

print("🚀 INIZIO GENERAZIONE MELODIE...")

# ==========================================
# 1. MELODIE PER ESTRAZIONE (Training)
# ==========================================
# Devono essere pulitissime e standard.
print("\n--- Generazione Set ESTRAZIONE (Scale e Arpeggi) ---")

# A. Scala Maggiore (La base assoluta)
save_sequence(
    freqs=[C4, D4, E4, F4, G4, A4, B4, C5], 
    durations=[0.5]*8, 
    folder=EXT_DIR, name="01_scale_major"
)

# B. Arpeggio Lento (Do-Mi-Sol)
save_sequence(
    freqs=[C4, E4, G4, C5, G4, E4], 
    durations=[0.8]*6, 
    folder=EXT_DIR, name="02_arpeggio_slow"
)

# C. Random Walk (Imprevedibile ma neutro)
random.seed(42) # Per riproducibilità
rand_notes = [random.choice([C4, D4, E4, F4, G4, A4, B4]) for _ in range(12)]
save_sequence(
    freqs=rand_notes, 
    durations=[0.4]*12, 
    folder=EXT_DIR, name="03_random"
)

# D. Melodia Discendente
save_sequence(
    freqs=[C5, B4, A4, G4, F4, E4, D4, C4], 
    durations=[0.5]*8, 
    folder=EXT_DIR, name="04_scale_down"
)

# E. Pentatonica (Neutro/Positiva)
save_sequence(
    freqs=[C4, D4, E4, G4, A4, C5], 
    durations=[0.4]*6, 
    folder=EXT_DIR, name="05_pentatonic"
)


# ==========================================
# 2. MELODIE PER TEST (Inference)
# ==========================================
# Devono essere diverse ma comunque neutre (senza emozione forte).
print("\n--- Generazione Set TEST (Pattern Diversi) ---")

# A. Pulsazione Ritmica (Nota ripetuta)
save_sequence(
    freqs=[G4, G4, G4, G4, C5, C5, G4, G4], 
    durations=[0.25]*8, 
    folder=TEST_DIR, name="test_01_pulse_rhythm"
)

# B. Bassi Profondi (Neutro/Scuro)
low_notes = [f/2 for f in [C4, G4, C4, F4]] # Ottava sotto
save_sequence(
    freqs=low_notes, 
    durations=[1.0]*4, 
    folder=TEST_DIR, name="test_02_deep_bass"
)

# C. Acuti Cristallini (Neutro/Chiaro)
high_notes = [f*2 for f in [E4, G4, B4, E4]] # Ottava sopra
save_sequence(
    freqs=high_notes, 
    durations=[0.3]*4, 
    folder=TEST_DIR, name="test_03_high_ping"
)

# D. Staccato Veloce (Ritmo frenetico)
fast_notes = [C4, C4, D4, D4, E4, E4, G4, G4]
save_sequence(
    freqs=fast_notes, 
    durations=[0.15]*8, 
    folder=TEST_DIR, name="test_04_staccato"
)

# E. Pause e Silenzi (Spazio vuoto)
gap_notes = [C4, 0, E4, 0, G4, 0, C5, 0] # 0 = Silenzio
save_sequence(
    freqs=gap_notes, 
    durations=[0.5]*8, 
    folder=TEST_DIR, name="test_05_gapped"
)

print("\n✅ TUTTE LE MELODIE GENERATE CON SUCCESSO!")