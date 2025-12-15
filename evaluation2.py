import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from transformers import pipeline
import librosa
import warnings

# ==========================================
# 6. EVALUATION CLASS (CON BPM)
# ==========================================
class Evaluation:
    def __init__(self, audio_folder, output_dir, csv_filename, train_mode=False, label_pos="happy mood", label_neg="sad mood"):
        """
        Inizializza la classe Evaluation.
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

        
        try:
            self.classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
            print("Modello CLAP caricato.\n")
        except:
            print("⚠️ Errore caricamento CLAP (forse manca libreria?)")

    def _get_valence_score(self, audio_path):
        """Calcola score AI (-1 a +1)"""
        try:
            output = self.classifier(audio_path, candidate_labels=self.candidate_labels)
            score_pos = next(item['score'] for item in output if item['label'] == self.label_pos)
            score_neg = next(item['score'] for item in output if item['label'] == self.label_neg)
            return score_pos - score_neg
        except Exception as e:
            return 0.0

    def extract_acoustic_features(self, audio_path):
        """
        Estrae metriche fisiche dal file audio usando Librosa.
        Ritorna un dizionario con i valori medi: CENTROIDE e BPM.
        """
        if not os.path.exists(audio_path):
            return {"centroid": 0, "bpm": 0}

        try:
            # Carica audio (solo primi 10s per velocità)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(audio_path, duration=10)
            
            # 1. Spectral Centroid (Brillantezza)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            avg_centroid = np.mean(cent)

            # 2. BPM (Tempo) - NUOVO CODICE
            # beat_track ritorna (tempo, beats). A noi serve solo tempo.
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # A volte librosa ritorna un array di 1 elemento, a volte un float. Normalizziamo.
            if isinstance(tempo, np.ndarray):
                tempo = tempo.item()
            
            return {
                "centroid": round(float(avg_centroid), 2),
                "bpm": round(float(tempo), 2)
            }
            
        except Exception as e:
            # In caso di errore (es. file troppo breve o silenzioso)
            return {"centroid": 0, "bpm": 0}

    def _create_bar_chart(self, x_labels, values, title, y_label, y_limit=(-1.1, 1.1)):
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(x_labels)) 
        x_labels_str = [str(x) for x in x_labels]
        
        cmap = plt.get_cmap('bwr')
        norm = mcolors.Normalize(vmin=-1, vmax=1) 
        bar_colors = cmap(norm(values))

        bars = plt.bar(x_pos, values, color=bar_colors, edgecolor='black', width=0.6)
        
        plt.axhline(0, color='black', linewidth=1.5)
        plt.ylim(y_limit[0], y_limit[1])
        
        plt.title(title, fontsize=16)
        
        plt.xlabel("Tracks ID", fontsize=12)  
        plt.ylabel(y_label, fontsize=12)      

        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.xticks(x_pos, x_labels_str, rotation=0) 

        y_range = y_limit[1] - y_limit[0]
        offset = y_range * 0.02

        for bar, score in zip(bars, values):
            y_pos = score + offset if score > 0 else score - (offset * 2)
            plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{score:.2f}", 
                     ha='center', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        
        if self.output_dir:
            safe_name = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
            save_path = os.path.join(self.output_dir, f"plot_{safe_name}.png")
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"📊 Grafico salvato: {save_path}")
        
        plt.show()
        plt.close()

    # --- PLOTTING FUNCTIONS ---
    def plot_neutral(self, num_samples=20, y_label="Score"):
        self._run_single_plot(num_samples, "_orig.wav", "Neutral Analysis", y_label)

    def plot_positive(self, num_samples=20, y_label="Score"):
        self._run_single_plot(num_samples, "_pos.wav", "Positive Analysis", y_label)

    def plot_negative(self, num_samples=20, y_label="Score"):
        self._run_single_plot(num_samples, "_neg.wav", "Negative Analysis", y_label)

    def _run_single_plot(self, num_samples, suffix, title, y_label):
        scores, labels = [], []
        for audio_id in self.ids[:num_samples]:
            path = os.path.join(self.audio_folder, f"{audio_id}{suffix}")
            if os.path.exists(path):
                scores.append(self._get_valence_score(path))
            else:
                scores.append(0)
            labels.append(audio_id)
        self._create_bar_chart(labels, scores, title, y_label)

    def plot_delta_positive(self, num_samples=20, y_label="Score"):
        self._run_delta_plot(num_samples, "_pos.wav", "Delta Positive (Pos - Orig)", y_label)

    def plot_delta_negative(self, num_samples=20, y_label="Score"):
        self._run_delta_plot(num_samples, "_neg.wav", "Delta Negative (Neg - Orig)", y_label)

    def _run_delta_plot(self, num_samples, target_suffix, title, y_label):
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
            labels.append(audio_id)
        
        self._create_bar_chart(labels, deltas, title, y_label, y_limit=(-2.0, 2.0))

    # --- SALVATAGGIO CSV  ---
    def save_to_csv(self):
        """Salva il CSV includendo metriche fisiche (CENTROIDE E BPM) e riga AVG finale."""
        os.makedirs(self.output_dir, exist_ok=True)
        full_path = os.path.join(self.output_dir, self.csv_filename)

        print(f"\n--- Inizio calcolo features e salvataggio CSV su {len(self.ids)} file ---")
        
        data = []

        for i, audio_id in enumerate(self.ids):
            path_pos = os.path.join(self.audio_folder, f"{audio_id}_pos.wav")
            path_neg = os.path.join(self.audio_folder, f"{audio_id}_neg.wav")
            
            s_pos = self._get_valence_score(path_pos) if os.path.exists(path_pos) else 0.0
            s_neg = self._get_valence_score(path_neg) if os.path.exists(path_neg) else 0.0

            # Estrai features (ora include anche BPM)
            feat_pos = self.extract_acoustic_features(path_pos)
            feat_neg = self.extract_acoustic_features(path_neg)

            if self.train_mode:
                data.append({
                    "id": audio_id,
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4),
                    "pos_centroid": feat_pos['centroid'],
                    "neg_centroid": feat_neg['centroid'],
                    "pos_bpm": feat_pos['bpm'],
                    "neg_bpm": feat_neg['bpm']
                })
            else:
                path_orig = os.path.join(self.audio_folder, f"{audio_id}_orig.wav")
                s_neutral = self._get_valence_score(path_orig) if os.path.exists(path_orig) else 0.0
                feat_orig = self.extract_acoustic_features(path_orig)
                
                data.append({
                    "id": audio_id,
                    "score_neutral": round(s_neutral, 4),
                    "score_pos": round(s_pos, 4),
                    "score_neg": round(s_neg, 4),
                    "delta_pos": round(s_pos - s_neutral, 4),
                    "delta_neg": round(s_neg - s_neutral, 4),
                    "orig_centroid": feat_orig['centroid'],
                    "pos_centroid": feat_pos['centroid'],
                    "neg_centroid": feat_neg['centroid'],
                    "orig_bpm": feat_orig['bpm'],
                    "pos_bpm": feat_pos['bpm'],
                    "neg_bpm": feat_neg['bpm']
                })

            if i % 5 == 0:
                print(f"Processati {i}/{len(self.ids)} file...")

        # Creazione DataFrame
        df = pd.DataFrame(data)
        
        # --- CALCOLO E STAMPA MEDIA ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols: numeric_cols.remove('id')
        
        avg_row = {col: round(df[col].mean(), 4) for col in numeric_cols}
        
        # === STAMPA MEDIE ===
        print("\n" + "="*40)
        print("📊  MEDIA PUNTEGGI (AVERAGE SCORES & METRICS)")
        print("="*40)
        if 'score_pos' in avg_row:
            print(f"🔹 Positive Score Avg: {avg_row['score_pos']}")
        if 'score_neg' in avg_row:
            print(f"🔸 Negative Score Avg: {avg_row['score_neg']}")
        
        if 'pos_bpm' in avg_row and 'neg_bpm' in avg_row:
            print(f"🥁 Positive BPM Avg: {avg_row['pos_bpm']}")
            print(f"🥁 Negative BPM Avg: {avg_row['neg_bpm']}")

        # Stampa anche i delta se siamo in FULL MODE
        if 'delta_pos' in avg_row:
            print(f"🔺 Delta Positive Avg: {avg_row['delta_pos']}")
        if 'delta_neg' in avg_row:
            print(f"🔻 Delta Negative Avg: {avg_row['delta_neg']}")
        print("="*40 + "\n")
        # =====================

        avg_row['id'] = "AVG" 
        df_avg = pd.DataFrame([avg_row])
        df = pd.concat([df, df_avg], ignore_index=True)
        
        try:
            df.to_csv(full_path, sep=';', index=False)
            print(f"✅ CSV completato e salvato in: {full_path}")
        except Exception as e:
            print(f"Errore nel salvare il CSV: {e}")

    # --- ORCHESTRATORE ---
    def evaluate(self, num_samples_plot=20, y_label="Score"):
        if self.train_mode:
            print("\n>>> TRAIN MODE <<<")
            self.plot_positive(num_samples_plot, y_label=y_label)
            self.plot_negative(num_samples_plot, y_label=y_label)
        else:
            print("\n>>> FULL MODE <<<")
            self.plot_neutral(num_samples_plot, y_label=y_label)
            self.plot_positive(num_samples_plot, y_label=y_label)
            self.plot_negative(num_samples_plot, y_label=y_label)
            self.plot_delta_positive(num_samples_plot, y_label=y_label)
            self.plot_delta_negative(num_samples_plot, y_label=y_label)
        self.save_to_csv()

    @classmethod
    def run(cls, audio_folder, output_dir, csv_filename, train_mode=False, num_samples=20, y_label="Score", label_pos="happy mood", label_neg="sad mood"):
        evaluator = cls(audio_folder, output_dir, csv_filename, train_mode, label_pos, label_neg)
        evaluator.evaluate(num_samples_plot=num_samples, y_label=y_label)
        return evaluator