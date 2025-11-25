@echo off
echo ===================================================
echo   SETUP AUTOMATICO PROGETTO AML (MusicGen Steering)
echo ===================================================
echo.

:: 1. Controllo FFmpeg
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [ATTENZIONE] FFmpeg non trovato!
    echo Per favore, apri un altro terminale come Amministratore e scrivi:
    echo winget install "FFmpeg (Essentials Build)"
    echo Poi riavvia questo script.
    pause
    exit /b
) else (
    echo [OK] FFmpeg rilevato.
)

:: 2. Installazione PyTorch (Versione Stabile CUDA 12.1)
echo.
echo [1/3] Installazione PyTorch con supporto GPU (CUDA)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: 3. Installazione Librerie di Supporto
echo.
echo [2/3] Installazione dipendenze da requirements.txt...
pip install -r requirements.txt

:: 4. Installazione AudioCraft (Modalita Sicura)
echo.
echo [3/3] Installazione AudioCraft (No-Deps Mode)...
:: Usiamo --no-deps per evitare che audiocraft provi a disinstallare il nostro PyTorch nuovo
pip install audiocraft --no-deps

echo.
echo ===================================================
echo   INSTALLAZIONE COMPLETATA CON SUCCESSO!
echo ===================================================
echo Ora prova a eseguire: python project_base.py
pause