# Setup Guide: Emotion Control in MusicGen

Questa guida serve per configurare l'ambiente di sviluppo su Windows per il progetto di AML.
Il setup gestisce automaticamente le dipendenze complesse di PyTorch (CUDA) e AudioCraft.

Per attivare il venv eseguire ogni volta:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\activate


### Prerequisiti
Prima di iniziare, assicurati di avere:
1. Python 3.9 o superiore installato.
2. FFmpeg (Fondamentale per salvare i file audio).
    * Verifica: Apri un terminale e scrivi ffmpeg -version.
    * Se dà errore: Apri PowerShell come Amministratore e lancia:
        winget install "FFmpeg (Essentials Build)"
    * Importante: Dopo l'installazione, riavvia il PC o chiudi/riapri completamente VS Code.

---

### Installazione (Step-by-Step)

Segui questi passaggi nell'ordine esatto per evitare conflitti di versione.

#### 1. Crea l'ambiente virtuale (Venv)
È obbligatorio usare un virtual environment per non rompere le installazioni globali di Python.
Apri il terminale nella cartella del progetto ed esegui:

python -m venv venv

#### 2. Attiva l'ambiente
Devi attivarlo ogni volta che inizi a lavorare.

* Comando Windows:
    .\venv\Scripts\activate

> Verifica: Se vedi la scritta (venv) all'inizio della riga del terminale, sei dentro.

#### 3. Esegui lo script di installazione automatica
Abbiamo creato uno script che installa PyTorch con supporto GPU e tutte le librerie nell'ordine corretto, ignorando i conflitti noti su Windows.

Con il venv attivo, esegui:

setup/setup_env.bat

*Attendi la fine del processo (potrebbe volerci qualche minuto per scaricare PyTorch).*

---

### Verifica del Funzionamento

Per essere sicuro che tutto funzioni (e che la GPU sia rilevata), esegui lo script base:

python project_base.py

Output atteso:
Dovresti vedere un messaggio verde "SUCCESSO! Il bypass funziona." e trovare un file audio generato nella cartella.

---

### Risoluzione Problemi Comuni

1. Errore "xformers not found" o Warning rossi
* È normale. Abbiamo disabilitato xformers perché su Windows causa crash. Lo script project_base.py gestisce automaticamente questo problema. Ignora i warning rossi nella console.

2. Errore "FileNotFoundError: ffmpeg"
* Significa che non hai installato FFmpeg o non hai riavviato il terminale dopo averlo fatto. Vedi la sezione "Prerequisiti".

3. La generazione è lenta
* Assicurati che project_base.py stampi Device: cuda. Se dice cpu, significa che non hai una scheda video NVIDIA configurata correttamente o non hai installato i driver CUDA.

---

### File Importanti

* project_base.py: Il "wrapper" principale. Importa sempre MusicGenWrapper da qui nei tuoi script (es. from project_base import MusicGenWrapper).
* setup_env.bat: Lo script che installa tutto.
* requirements.txt: La lista delle librerie (non modificarlo manualmente).
