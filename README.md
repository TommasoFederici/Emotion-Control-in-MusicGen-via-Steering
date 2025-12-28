# Emotion Control in MusicGen via Activation Steering

**Authors:** Lucia Fornetti & Tommaso Federici

Questo repository contiene il codice ufficiale per il progetto **"Emotion control in MusicGen via Activation Steering"**.
Il progetto implementa un framework leggero per controllare attributi emotivi di alto livello (es. "Happy" vs "Sad") in **MusicGen** utilizzando la tecnica dell'**Activation Steering**, eliminando la necessità di un fine-tuning costoso del modello.

## 🎵 Caratteristiche Principali

* **Zero Fine-Tuning:** Il controllo emotivo avviene interamente a tempo di inferenza iniettando vettori di steering nelle attivazioni interne del modello.
* **Strategia Multi-Block:** Utilizziamo un approccio innovativo che inietta vettori diversi in blocchi specifici del Transformer per controllare separatamente caratteristiche ritmiche (Mid-Block, Layer 12) e timbriche (Deep-Block, Layer 30).
* **Alpha Decay:** Implementazione di un meccanismo di decadimento del coefficiente di steering ($\gamma = 0.998$) per mantenere la coerenza strutturale dell'audio ed evitare artefatti.
* **Efficiente:** Basato sul checkpoint `musicgen-melody` (utilizzato in modalità text-only).

## 📂 Struttura della Repository

* `Emotion_Control_ActivationSteering_demo_code_colab.ipynb`: **Notebook principale**. Contiene tutto il codice necessario per caricare il modello, estrarre i vettori (o caricarli) e generare musica controllata. È pronto per l'uso su Google Colab.
* `core_colab_melody.py`: Contiene la logica core del progetto, incluse le classi per l'hooking del modello MusicGen e l'implementazione dello steering.
* `data/`:
    * `Happy_Sad/`: Dataset di prompt utilizzati per l'estrazione dei vettori (`extraction.csv`) e per i test (`inference.csv`).
    * `vectors/`: Contiene i tensori pre-calcolati (`steering_vectors.pt`) per l'emozione Happy/Sad, permettendo l'inferenza immediata senza dover rieseguire l'estrazione.
* `server_blind_evaluation/`: Codice (Python/HTML) utilizzato per condurre il *blind listening test* descritto nel report per la validazione umana dei risultati.

## 🚀 Quick Start (Come usare)

Il modo più semplice per provare il modello è utilizzare il notebook fornito:

1.  Apri il file `Emotion_Control_ActivationSteering_demo_code_colab.ipynb` (consigliato l'uso di Google Colab con GPU T4).
2.  Installa le dipendenze richieste (eseguendo la prima cella).
3.  Carica i vettori pre-calcolati da `data/vectors/steering_vectors.pt` oppure esegui la fase di estrazione sui tuoi prompt.
4.  Esegui la generazione modificando il parametro `steering_strength` (valori positivi per "Happy", negativi per "Sad").

## 🧠 Metodologia in Breve

Il sistema interviene direttamente sul residual stream del Transformer durante la generazione autoregressiva. Abbiamo identificato due punti di intervento ottimali tramite analisi della *Silhouette Score*:

1.  **Mid-Block (Layer 11-14):** Controlla feature a basso livello come tempo e brillantezza.
2.  **Deep-Block (Layer 27-29):** Controlla feature timbriche e di texture.

L'intensità dello steering decade nel tempo secondo la formula $\alpha(t) = \alpha_0 \cdot \gamma^t$ per garantire transizioni naturali e prevenire la saturazione del segnale.