import os
import csv
import datetime
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# --- CONFIGURAZIONE PERCORSI ASSOLUTI ---
# 1. Trova la cartella dove si trova QUESTO file server.py (es. .../Emotion-Control-in-MusicGen-via-Steering)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Definisce i percorsi esatti per gli altri file nella stessa cartella
HTML_FILE = os.path.join(BASE_DIR, 'index.html')
AUDIO_FOLDER = r"C:\Users\tomma\OneDrive\Documenti\Git Projects\Emotion-Control-in-MusicGen-via-Steering\server_blind_evaluation\blind_evaluation_test_audio"
CSV_FILE = os.path.join(BASE_DIR, 'risultati_test.csv')

# --- INIZIALIZZAZIONE ---

# Crea la cartella audio se non esiste
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)
    print(f"Ho creato la cartella: {AUDIO_FOLDER}")

# Crea il CSV se non esiste
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["User_ID", "Gruppo", "File_Reale", "Emozione_Scelta"])

# --- ROTTE DEL SITO ---

@app.route('/')
def home():
    # LEGGE IL FILE HTML USANDO IL PERCORSO COMPLETO
    try:
        with open(HTML_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"<h1>Errore Grave</h1><p>Non trovo il file: {HTML_FILE}</p><p>Assicurati che index.html sia nella stessa cartella di server.py</p>", 404

@app.route('/lista-audio')
def list_files():
    try:
        # Legge solo file audio
        files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(('.mp3', '.wav', '.ogg'))]
        return jsonify(files)
    except Exception as e:
        print(f"Errore lettura audio: {e}")
        return jsonify([])

@app.route('/audio/<path:filename>')
def get_audio(filename):
    # Serve il file audio al browser
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route('/salva-test', methods=['POST'])
def salva_test():
    data = request.json
    user_id = data.get('userId')
    risultati = data.get('risultati')
    
    if not user_id or not risultati:
        return jsonify({"status": "errore", "msg": "Dati mancanti"}), 400

    ora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for res in risultati:
                writer.writerow([
                    user_id, 
                    ora, 
                    res['gruppo'], 
                    res['file_reale'], 
                    res['emozione']
                ])
        return jsonify({"status": "successo", "msg": "Test salvato!"})
    except Exception as e:
        print(f"Errore salvataggio CSV: {e}")
        return jsonify({"status": "errore", "msg": "Errore scrittura file"}), 500

if __name__ == '__main__':
    print("------------------------------------------------")
    print(f"Server avviato dalla cartella: {BASE_DIR}")
    print(f"Sto cercando index.html in: {HTML_FILE}")
    print("Vai su http://localhost:5000")
    print("------------------------------------------------")
    app.run(host='0.0.0.0', port=5000) 