import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

def analyze_and_plot():
    # --- CONFIGURAZIONE ---
    # Inserisci qui i tuoi percorsi esatti
    base_dir = r"C:\Users\lucia\Desktop\Uni2\AML\Progetto_finale\Emotion-Control-in-MusicGen-via-Steering\server_blind_evaluation"
    input_csv = os.path.join(base_dir, "risultati_test.csv")
    output_scores_csv = os.path.join(base_dir, "score_finali.csv")
    output_img_dir = base_dir

    # Mappa dei punteggi
    score_map = { "Happy": 1, "Neutral": 0, "Sad": -1 }

    if not os.path.exists(input_csv):
        print(f"❌ Errore: Il file {input_csv} non esiste.")
        return

    print(f"📂 Lettura dati da: {input_csv}...")
    
    try:
        # 1. LETTURA ROBUSTA DEL CSV
        try:
            df = pd.read_csv(input_csv, sep=',', skipinitialspace=True)
        except:
            df = pd.read_csv(input_csv, sep=',', on_bad_lines='skip', skipinitialspace=True)

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        
        # 2. RINOMINA COLONNE
        rename_map = {
            'user_answer': 'User_Choice',
            'trak_name': 'Filename',
            'tracks_groups': 'Track_ID_Ref',
            'user_id': 'User_ID'
        }
        df.rename(columns=rename_map, inplace=True)

        if 'User_Choice' not in df.columns:
            print("❌ ERRORE: Colonna 'user_answer' non trovata.")
            return

        # 3. CALCOLO PUNTEGGI
        df['User_Choice'] = df['User_Choice'].astype(str).str.strip()
        df['Numeric_Score'] = df['User_Choice'].map(score_map).fillna(0).astype(int)

        # Raggruppa per ID traccia e Nome File per ottenere la media per ogni file
        grouped = df.groupby(['Track_ID_Ref', 'Filename']).agg(
            Average_Score=('Numeric_Score', 'mean')
        ).reset_index()

        # Assicura che ID sia numerico per l'ordinamento
        grouped['Track_ID_Ref'] = pd.to_numeric(grouped['Track_ID_Ref'], errors='coerce')
        grouped = grouped.sort_values(by=['Track_ID_Ref'])

        # 4. CALCOLO DELTA E STRUTTURAZIONE DATI
        def get_category(filename):
            s = str(filename).lower().strip()
            if "_pos" in s: return "Positive"
            if "_neg" in s: return "Negative"
            if "_orig" in s: return "Neutral"
            return "Unknown"

        grouped['Category'] = grouped['Filename'].apply(get_category)

        # Creazione Pivot Table: Una riga per Track ID, colonne per le categorie
        pivot_df = grouped.pivot_table(index='Track_ID_Ref', columns='Category', values='Average_Score')

        # Assicura che tutte le colonne esistano
        for col in ['Positive', 'Negative', 'Neutral']:
            if col not in pivot_df.columns:
                pivot_df[col] = np.nan

        # Calcolo dei DELTA
        pivot_df['Delta_Pos'] = pivot_df['Positive'] - pivot_df['Neutral']
        pivot_df['Delta_Neg'] = pivot_df['Negative'] - pivot_df['Neutral']

        # Calcolo MEDIE totali
        averages = pivot_df.mean()
        avg_row = pd.DataFrame(averages).T
        avg_row.index = ['AVG']
        
        final_df = pd.concat([pivot_df, avg_row])

        # Salva CSV
        final_df.to_csv(output_scores_csv, sep=',', index=True, float_format='%.3f')
        print(f"✅ File CSV aggiornato con Delta e Medie salvato in: {output_scores_csv}")

        # 5. GENERAZIONE GRAFICI
        plot_data = pivot_df.copy()
        ids = plot_data.index.astype(str).tolist()

        # Helper aggiornato per accettare limiti custom
        def plot_column_safe(col_name, title, suffix, custom_limit):
            if col_name in plot_data.columns:
                values = plot_data[col_name].fillna(0).tolist()
                create_bar_chart(ids, values, title, output_img_dir, suffix, y_lim=custom_limit)
            else:
                print(f"⚠️ Dati mancanti per: {col_name}")

        # --- GRAFICI SCORE (Range -1 a 1) ---
        plot_column_safe('Positive', "Positive Tracks Evaluation", "positive", 1.0)
        plot_column_safe('Negative', "Negative Tracks Evaluation", "negative", 1.0)
        plot_column_safe('Neutral', "Neutral Tracks Evaluation", "neutral", 1.0)
        
        # --- GRAFICI DELTA (Range -2 a 2) ---
        plot_column_safe('Delta_Pos', "Delta Positive (Pos - Neutral)", "delta_pos", 2.0)
        plot_column_safe('Delta_Neg', "Delta Negative (Neg - Neutral)", "delta_neg", 2.0)

    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

def create_bar_chart(x_labels, values, title, output_dir, filename_suffix, y_lim):
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(x_labels))
    
    # Colori: Blu(Negativo) -> Bianco(0) -> Rosso(Positivo)
    cmap = plt.get_cmap('bwr')
    
    # La normalizzazione colore segue i limiti del grafico
    # Se y_lim è 2, il rosso massimo sarà a +2. Se è 1, sarà a +1.
    norm = mcolors.Normalize(vmin=-y_lim, vmax=y_lim) 
    bar_colors = cmap(norm(values))

    bars = plt.bar(x_pos, values, color=bar_colors, edgecolor='black', width=0.6)
    
    plt.axhline(0, color='black', linewidth=1.5)
    
    # Imposta i limiti dell'asse Y in base al parametro passato
    plt.ylim(-y_lim, y_lim)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Track ID", fontsize=12)
    plt.ylabel("Score / Delta", fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(x_pos, x_labels, rotation=0)

    # Etichette numeriche
    offset = y_lim * 0.05
    for bar, score in zip(bars, values):
        # Evita che l'etichetta esca dal grafico
        draw_y = score
        if score > y_lim: draw_y = y_lim - offset
        if score < -y_lim: draw_y = -y_lim + offset
        
        y_txt = draw_y + offset if score >= 0 else draw_y - offset
        
        plt.text(bar.get_x() + bar.get_width()/2, y_txt, f"{score:.2f}", 
                ha='center', va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, f"plot_blind_{filename_suffix}.png")
        plt.savefig(save_path, dpi=300)
        print(f"📊 Grafico salvato: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    analyze_and_plot()