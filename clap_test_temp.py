from core import MusicGenWrapper
from transformers import pipeline

audio_tr = "audio_triste"
audio_fe = "audio_felice"

mg = MusicGenWrapper(size = 'small', duration=5)

n = 0
'''prompt = ["happy song", "cheerful song to dance", "very happy piano solo", "warm cristmas song happy mood", "dark depressive winter song", "sad song", "depressive music to cry", "very sad piano solo"]
for p in prompt:
    if n < 4:
        nome = f"songs/happy/{n}_H"
    else:
        nome = f"songs/sad/{n}_S"
    mg.generate(p, nome)
    n = n+1'''

################################################
################### CLAP TEST ##################
################################################
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from transformers import pipeline

# Configurazione Etichette
label_happy = "happy mood" # Etichetta rinforzata
label_sad = "sad mood"  # Etichetta rinforzata
labels = [label_happy, label_sad]

print("\nCaricamento modello CLAP...")
audio_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")

def get_valence_score(audio_path):
    """
    Restituisce un valore tra -1 (Triste) e +1 (Felice)
    """
    try:
        output = audio_classifier(audio_path, candidate_labels=labels)
        
        score_happy = next(item['score'] for item in output if item['label'] == label_happy)
        score_sad = next(item['score'] for item in output if item['label'] == label_sad)
        
        print(f"Analizzato {audio_path} -> Happy: {score_happy:.3f} | Sad: {score_sad:.3f}")
        
        # Formula: Happy - Sad
        return score_happy - score_sad
    except Exception as e:
        print(f"Errore nel caricare il file {audio_path}: {e}")
        return 0 # Ritorna neutro se file non trovato

files_felici = []
files_tristi = []

for i in range(4):
    path = f"songs/happy/{i}_H.wav" 
    files_felici.append(path)

for i in range(4, 8):
    path = f"songs/sad/{i}_S.wav"
    files_tristi.append(path)

print(f"\nFile felici trovati: {files_felici}")
print(f"File tristi trovati: {files_tristi}")

print("\nInizio calcolo score...")
scores_felici = [get_valence_score(f) for f in files_felici]
scores_tristi = [get_valence_score(f) for f in files_tristi]

all_scores = scores_felici + scores_tristi

labels_felici = [f"Happy_{i}" for i in range(4)]
labels_tristi = [f"Sad_{i}" for i in range(4, 8)]
all_labels = labels_felici + labels_tristi



plt.figure(figsize=(12, 7)) 

cmap = plt.get_cmap('bwr') 
norm = mcolors.Normalize(vmin=-1, vmax=1) 

bar_colors = cmap(norm(all_scores))

bars = plt.bar(all_labels, all_scores, color=bar_colors, edgecolor='black', width=0.6)

plt.axhline(0, color='black', linewidth=1.5)
plt.ylim(-1.1, 1.1)
plt.title('Emotion Analysis', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.5)


for bar, score in zip(bars, all_scores):
    y_pos = score + 0.05 if score > 0 else score - 0.10
    plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{score:.2f}", 
             ha='center', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()