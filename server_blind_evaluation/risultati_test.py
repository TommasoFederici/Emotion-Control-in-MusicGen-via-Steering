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
        # Il tuo file ha spazi dopo la virgola (es. ", date") -> skipinitialspace=True
        # Il tuo file ha una virgola finale nella header -> engine='python' gestisce meglio errori di parsing
        try:
            df = pd.read_csv(input_csv, sep=',', skipinitialspace=True)
        except:
            # Fallback se ci sono problemi strani di righe
            df = pd.read_csv(input_csv, sep=',', on_bad_lines='skip', skipinitialspace=True)

        # Rimuove eventuali colonne vuote create dalla virgola finale (es. "Unnamed: 5")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Pulisce i nomi delle colonne da spazi extra
        df.columns = df.columns.str.strip()
        
        print(f"📋 Colonne trovate: {df.columns.tolist()}")

        # 2. RINOMINA COLONNE (Standardizzazione)
        # Mappiamo i TUOI nomi (dal csv che mi hai mandato) ai nomi dello script
        rename_map = {
            'user_answer': 'User_Choice',
            'trak_name': 'Filename',
            'tracks_groups': 'Track_ID_Ref',
            'user_id': 'User_ID'
        }
        df.rename(columns=rename_map, inplace=True)

        # Controllo di sicurezza
        if 'User_Choice' not in df.columns:
            print("❌ ERRORE: Non trovo la colonna delle risposte (es. 'user_answer').")
            print(f"   Colonne attuali: {df.columns.tolist()}")
            return

        # 3. CALCOLO PUNTEGGI
        # Converte Happy->1, Sad->-1, ecc.
        # .str.strip() è fondamentale perché nel CSV potresti avere "Sad " con spazi
        df['User_Choice'] = df['User_Choice'].astype(str).str.strip()
        df['Numeric_Score'] = df['User_Choice'].map(score_map).fillna(0).astype(int)

        # Raggruppa per ID traccia e Nome File
        grouped = df.groupby(['Track_ID_Ref', 'Filename']).agg(
            Total_Score=('Numeric_Score', 'sum'),
            Num_Votes=('User_ID', 'count')
        ).reset_index()

        # Ordina per ID
        grouped['Track_ID_Ref'] = pd.to_numeric(grouped['Track_ID_Ref'], errors='coerce')
        grouped = grouped.sort_values(by=['Track_ID_Ref'])
        
        # Salva CSV dei punteggi
        grouped.to_csv(output_scores_csv, index=False)
        print(f"✅ Punteggi salvati in: {output_scores_csv}")

        # 4. CATEGORIE E PLOT
        def get_category(filename):
            s = str(filename).lower().strip()
            if "_pos" in s: return "Positive"
            if "_neg" in s: return "Negative"
            if "_orig" in s: return "Neutral"
            return "Unknown"

        grouped['Category'] = grouped['Filename'].apply(get_category)

        # Calcola scala grafico
        max_abs_score = grouped['Total_Score'].abs().max()
        if pd.isna(max_abs_score) or max_abs_score == 0: max_abs_score = 1
        y_limit = max_abs_score + 1

        categories = ["Positive", "Negative", "Neutral"]
        
        for cat in categories:
            subset = grouped[grouped['Category'] == cat]
            if subset.empty:
                print(f"⚠️ Nessun dato per la categoria: {cat}")
                continue

            ids = subset['Track_ID_Ref'].astype(str).tolist()
            scores = subset['Total_Score'].tolist()
            
            # Titoli
            if cat == "Positive": title = "Valutazione Tracce POSITIVE (Target: Happy)"
            elif cat == "Negative": title = "Valutazione Tracce NEGATIVE (Target: Sad)"
            else: title = "Valutazione Tracce ORIGINALI (Target: Neutral)"

            create_bar_chart(ids, scores, title, output_img_dir, cat.lower(), max_abs_score, y_limit)

    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

def create_bar_chart(x_labels, values, title, output_dir, filename_suffix, max_val, y_lim):
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(x_labels))
    
    # Colori: Blu(Negativo) -> Bianco(0) -> Rosso(Positivo)
    cmap = plt.get_cmap('bwr')
    norm = mcolors.Normalize(vmin=-max_val, vmax=max_val) 
    bar_colors = cmap(norm(values))

    bars = plt.bar(x_pos, values, color=bar_colors, edgecolor='black', width=0.6)
    
    plt.axhline(0, color='black', linewidth=1.5)
    plt.ylim(-y_lim, y_lim)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Track ID", fontsize=12)
    plt.ylabel("Total Score", fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(x_pos, x_labels, rotation=0)

    # Etichette numeriche sulle barre
    offset = y_lim * 0.05
    for bar, score in zip(bars, values):
        y_txt = score + offset if score >= 0 else score - offset
        plt.text(bar.get_x() + bar.get_width()/2, y_txt, f"{int(score)}", 
                 ha='center', va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, f"plot_blind_{filename_suffix}.png")
        plt.savefig(save_path, dpi=300)
        print(f"📊 Grafico salvato: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    analyze_and_plot()