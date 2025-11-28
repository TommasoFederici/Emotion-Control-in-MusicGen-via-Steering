import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from transformers import pipeline

######## FILE DA ELIMINARE (POI):
# score_test_Happy_Sad_TRAIN.csv
# score_test_Happy_Sad.csv
# audio
# clap_test_temp.py


class Evaluation:
    def __init__(self, audio_folder, output_dir, csv_filename, train_mode=False, label_pos="happy mood", label_neg="sad mood"):
        """
        Inizializza la classe Evaluation.
        Args:
            audio_folder (str): Cartella con i file audio.
            output_dir (str): Cartella dove salvare il CSV.
            csv_filename (str): Nome del file CSV finale.
            train_mode (bool): Se True, calcola solo Pos/Neg. Se False, calcola anche Orig/Delta.
            label_pos (str): Etichetta positiva.
            label_neg (str): Etichetta negativa.
        """
        self.audio_folder = audio_folder
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        self.train_mode = train_mode
        
        self.label_pos = label_pos
        self.label_neg = label_neg
        self.candidate_labels = [self.label_pos, self.label_neg]
        self.ids = []

        print(f"--- Inizializzazione Evaluation (Mode: {'TRAIN' if train_mode else 'FULL EVAL'}) ---")
        
        # Scansione ID
        # Se siamo in TRAIN, potremmo non avere i file _orig, quindi scansioniamo i _pos
        scan_suffix = "_pos.wav" if self.train_mode else "_orig.wav"
        
        try:
            files = os.listdir(self.audio_folder)
            for f in files:
                if f.endswith(scan_suffix): 
                    try:
                        self.ids.append(int(f.split('_')[0]))
                    except ValueError: pass
            self.ids.sort()
            print(f"File trovati (suffix '{scan_suffix}'): {len(self.ids)} ID unici.")
        except FileNotFoundError:
            print(f"ERRORE: Cartella {self.audio_folder} non trovata.")

        
        self.classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
        print("Modello CLAP caricato.\n")

    def _get_valence_score(self, audio_path):
        """Calcola score (-1 a +1)"""
        try:
            output = self.classifier(audio_path, candidate_labels=self.candidate_labels)
            score_pos = next(item['score'] for item in output if item['label'] == self.label_pos)
            score_neg = next(item['score'] for item in output if item['label'] == self.label_neg)
            return score_pos - score_neg
        except Exception as e:
            return 0.0

    def _create_bar_chart(self, x_labels, values, title, y_limit=(-1.1, 1.1)):
        plt.figure(figsize=(12, 6))
        cmap = plt.get_cmap('bwr')
        norm = mcolors.Normalize(vmin=-1, vmax=1) 
        bar_colors = cmap(norm(values))

        bars = plt.bar(x_labels, values, color=bar_colors, edgecolor='black', width=0.6)
        plt.axhline(0, color='black', linewidth=1.5)
        plt.ylim(y_limit[0], y_limit[1])
        plt.title(title, fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)

        y_range = y_limit[1] - y_limit[0]
        offset = y_range * 0.02

        for bar, score in zip(bars, values):
            y_pos = score + offset if score > 0 else score - (offset * 2)
            plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{score:.2f}", 
                     ha='center', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.show()

    # --- PLOTTING FUNCTIONS ---
    def plot_neutral(self, num_samples=20):
        self._run_single_plot(num_samples, "_orig.wav", "Neutral Analysis")

    def plot_positive(self, num_samples=20):
        self._run_single_plot(num_samples, "_pos.wav", "Positive Analysis")

    def plot_negative(self, num_samples=20):
        self._run_single_plot(num_samples, "_neg.wav", "Negative Analysis")

    def _run_single_plot(self, num_samples, suffix, title):
        scores, labels = [], []
        for audio_id in self.ids[:num_samples]:
            path = os.path.join(self.audio_folder, f"{audio_id}{suffix}")
            if os.path.exists(path):
                scores.append(self._get_valence_score(path))
            else:
                scores.append(0)
            labels.append(f"ID_{audio_id}")
        self._create_bar_chart(labels, scores, title)

    def plot_delta_positive(self, num_samples=20):
        self._run_delta_plot(num_samples, "_pos.wav", "Delta Positive (Pos - Orig)")

    def plot_delta_negative(self, num_samples=20):
        self._run_delta_plot(num_samples, "_neg.wav", "Delta Negative (Neg - Orig)")

    def _run_delta_plot(self, num_samples, target_suffix, title):
        deltas, labels = [], []
        for audio_id in self.ids[:num_samples]:
            path_orig = os.path.join(self.audio_folder, f"{audio_id}_orig.wav")
            path_target = os.path.join(self.audio_folder, f"{audio_id}{target_suffix}")
            
            if os.path.exists(path_orig) and os.path.exists(path_target):
                s_orig = self._get_valence_score(path_orig)
                s_target = self._get_valence_score(path_target)
                deltas.append(s_target - s_orig)
            else:
                deltas.append(0)
            labels.append(f"ID_{audio_id}")
        
        self._create_bar_chart(labels, deltas, title, y_limit=(-2.0, 2.0))

    # --- SALVATAGGIO CSV ---
    def save_to_csv(self):
        """
        Salva il CSV in base alla modalità (TRAIN o FULL).
        """
        # Assicura che la cartella di output esista
        os.makedirs(self.output_dir, exist_ok=True)
        full_path = os.path.join(self.output_dir, self.csv_filename)

        print(f"\n--- Inizio calcolo e salvataggio CSV su {len(self.ids)} file ---")
        print(f"Modalità TRAIN: {self.train_mode}")
        
        data = []

        for i, audio_id in enumerate(self.ids):
            # Percorsi comuni
            path_pos = os.path.join(self.audio_folder, f"{audio_id}_pos.wav")
            path_neg = os.path.join(self.audio_folder, f"{audio_id}_neg.wav")
            
            # Calcolo score Pos/Neg
            s_pos = self._get_valence_score(path_pos) if os.path.exists(path_pos) else 0.0
            s_neg = self._get_valence_score(path_neg) if os.path.exists(path_neg) else 0.0

            if self.train_mode:
                # MODALITA' TRAIN: Solo Pos e Neg
                data.append({
                    "id": audio_id,
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4)
                })
            else:
                # MODALITA' DEFAULT: Anche Orig e Delta
                path_orig = os.path.join(self.audio_folder, f"{audio_id}_orig.wav")
                s_neutral = self._get_valence_score(path_orig) if os.path.exists(path_orig) else 0.0
                
                d_pos = s_pos - s_neutral
                d_neg = s_neg - s_neutral

                data.append({
                    "id": audio_id,
                    "score_neutral": round(s_neutral, 4),
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4),
                    "delta_pos": round(d_pos, 4),
                    "delta_neg": round(d_neg, 4)
                })

            if i % 5 == 0:
                print(f"Processati {i}/{len(self.ids)} file...")

        # Creazione DataFrame
        df = pd.DataFrame(data)
        
        # Ordinamento colonne
        if self.train_mode:
            cols = ["id", "score_pos", "score_neg"]
        else:
            cols = ["id", "score_neutral", "score_pos", "score_neg", "delta_pos", "delta_neg"]
        
        df = df[cols]
        
        try:
            df.to_csv(full_path, sep=';', index=False)
            print(f"\nCOMPLETATO. File salvato in: {full_path}")
            print(df.head()) 
        except Exception as e:
            print(f"Errore nel salvare il CSV: {e}")

    # --- FUNZIONE ORCHESTRATORE ---
    def evaluate(self, num_samples_plot=20):
        if self.train_mode:
            print("\n>>> TRAIN MODE <<<")
            self.plot_positive(num_samples_plot)
            self.plot_negative(num_samples_plot)
        else:
            print("\n>>> FULL MODE <<<")
            self.plot_neutral(num_samples_plot)
            self.plot_positive(num_samples_plot)
            self.plot_negative(num_samples_plot)
            self.plot_delta_positive(num_samples_plot)
            self.plot_delta_negative(num_samples_plot)
        self.save_to_csv()

    @classmethod
    def run(cls, audio_folder, output_dir, csv_filename, train_mode=False, num_samples=20):
        evaluator = cls(audio_folder, output_dir, csv_filename, train_mode)
        evaluator.evaluate(num_samples_plot=num_samples)
        return evaluator

# --- UTILIZZO SEMPLIFICATO ---
if __name__ == "__main__":
    
    Evaluation.run(
        audio_folder="data/Happy_Sad/audio",
        output_dir="data/Happy_Sad", 
        csv_filename="score_test_Happy_Sad_TRAIN.csv",
        train_mode=True,  
        #label_pos="calm mood",
        #label_neg="angry mood",
        num_samples=20
    )