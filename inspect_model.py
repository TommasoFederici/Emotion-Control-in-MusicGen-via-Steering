from audiocraft.models import MusicGen

model = MusicGen.get_pretrained('small')

# MusicGen ha diverse parti: 
# 1. compression_model (EnCodec - non ci interessa, tocca solo l'audio grezzo)
# 2. lm (Language Model - IL NOSTRO OBIETTIVO)

# 3. conditioners (Processano il testo)

print(model.lm)

# Se stampi questo, vedrai una struttura enorme. 
# Cerca qualcosa come 'transformer.layers'.
# Solitamente in PyTorch è: model.lm.transformer.layers