# ==========================================
# 4. DATASET EXTRACTOR (multi-layer)
# ==========================================
class DatasetExtractor:
    def __init__(self, model_wrapper, layers=[14]):
        """
        Args:
            layers: Può essere un int (14) o una lista ([12, 13, 14])
        """
        self.mg = model_wrapper
        
        if isinstance(layers, int): layers = [layers]
        self.target_layers_indices = layers
        
        # Dizionario di Hook: {layer_idx: HookInstance}
        self.hooks = {}
        for idx in self.target_layers_indices:
            layer_module = self.mg.model.lm.transformer.layers[idx]
            self.hooks[idx] = ActivationHook(layer_module)
        
    def extract(self, csv_path, save_path, audio_output_dir=None, sep=';'):
        print(f"🏭 Multi-Extraction {self.target_layers_indices} -> {save_path}")
        
        try: df = pd.read_csv(csv_path, sep=sep)
        except: print("❌ Error reading CSV"); return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if audio_output_dir: os.makedirs(audio_output_dir, exist_ok=True)

        # Accumulatori per ogni layer: {14: TensorAccumulatore, 15: ...}
        layer_sums = {idx: None for idx in self.target_layers_indices}
        count = 0

        # Attiva tutti gli hook
        for h in self.hooks.values(): h.register()

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                p_pos = str(row.get('positive_prompt', row.iloc[0])).strip()
                p_neg = str(row.get('negative_prompt', row.iloc[1])).strip()
                pid = str(row.get('ID', index)).strip()

                # --- GENERA POSITIVE ---
                f_pos = os.path.join(audio_output_dir, f"{pid}_pos") if audio_output_dir else None
                self.mg.generate(p_pos, filename=f_pos)
                vecs_pos = {idx: h.get_mean_vector() for idx, h in self.hooks.items()}
                for h in self.hooks.values(): h.activations = [] 

                # --- GENERA NEGATIVE ---
                f_neg = os.path.join(audio_output_dir, f"{pid}_neg") if audio_output_dir else None
                self.mg.generate(p_neg, filename=f_neg)
                vecs_neg = {idx: h.get_mean_vector() for idx, h in self.hooks.items()}
                for h in self.hooks.values(): h.activations = [] 

                # --- CALCOLA DIFFERENZE E ACCUMULA ---
                valid_pair = True
                current_diffs = {}
                
                for idx in self.target_layers_indices:
                    v_p = vecs_pos[idx]
                    v_n = vecs_neg[idx]
                    if v_p is None or v_n is None: 
                        valid_pair = False; break
                    
                    diff = v_p - v_n
                    # Normalizzazione Locale (Fondamentale)
                    diff = diff / (diff.norm() + 1e-8)
                    current_diffs[idx] = diff

                if valid_pair:
                    for idx, diff in current_diffs.items():
                        if layer_sums[idx] is None: layer_sums[idx] = diff
                        else: layer_sums[idx] += diff
                    count += 1

            except Exception as e: print(f"Err {index}: {e}")

        # Rimuovi hook
        for h in self.hooks.values(): h.remove()

        # --- CALCOLO VETTORE FINALE PER OGNI LAYER ---
        final_vectors_dict = {}
        
        if count > 0:
            print(f"🧮 Calcolo medie su {count} coppie...")
            for idx, total_sum in layer_sums.items():
                if total_sum is None: continue
                
                # Media Semplice
                mean_vec = total_sum / count
                
                # Normalizzazione Finale
                mean_vec = mean_vec / (mean_vec.norm() + 1e-8)
                if mean_vec.dim() == 1: mean_vec = mean_vec.unsqueeze(0)
                
                final_vectors_dict[idx] = mean_vec

            # Salviamo un DIZIONARIO {layer_idx: vector}
            torch.save(final_vectors_dict, save_path)
            print(f"✅ Multi-Layer Vector salvato: {save_path}")
            print(f"   Contiene layer: {list(final_vectors_dict.keys())}")
        else:
            print("❌ Nessun vettore estratto (count=0).")

# ==========================================
# 5. DATASET INFERENCE (multi-layer)
# ==========================================
class DatasetInference:
    def __init__(self, model_wrapper, layers=None):
        """
        layers: Lista di layer da USARE per l'inferenza.
                Se None, usa TUTTI i layer trovati nel file .pt.
        """
        self.mg = model_wrapper
        # Normalizza a lista
        if isinstance(layers, int): layers = [layers]
        self.filter_layers = layers

    def run(self, prompts_file, vector_path, output_dir, alpha=1.5, max_samples=None):
        print(f"🚀 Batch Inference (Alpha {alpha})...")
        
        steerers = []
        try:
            data = torch.load(vector_path)
            
            # Determina quali layer usare
            available_layers = []
            if isinstance(data, dict):
                available_layers = list(data.keys())
            else:
                available_layers = [14] # Fallback per vecchi file singoli

            # Se l'utente ha specificato dei layer, usiamo l'intersezione
            if self.filter_layers:
                target_layers = [l for l in self.filter_layers if l in available_layers]
                if len(target_layers) < len(self.filter_layers):
                    print(f"⚠️ Warning: Alcuni layer richiesti non sono nel file .pt. Userò: {target_layers}")
            else:
                target_layers = available_layers

            print(f"🎯 Steering attivo sui layer: {target_layers}")

            # Caricamento Steerers
            if isinstance(data, dict):
                for idx in target_layers:
                    vec = data[idx]
                    vec = vec / (vec.norm() + 1e-8)
                    if vec.dim() == 2 and vec.shape[0] > 1: vec = vec.mean(dim=0, keepdim=True)
                    
                    target_mod = self.mg.model.lm.transformer.layers[idx]
                    steerers.append(DynamicSteering(target_mod, vec))
            else:
                # Caso singolo (compatibilità)
                vec = data
                vec = vec / (vec.norm() + 1e-8)
                if vec.dim() == 2 and vec.shape[0] > 1: vec = vec.mean(dim=0, keepdim=True)
                for idx in target_layers:
                    target_mod = self.mg.model.lm.transformer.layers[idx]
                    steerers.append(DynamicSteering(target_mod, vec))
                    
        except Exception as e: print(f"❌ Error loading vector: {e}"); return

        if not steerers: print("❌ Nessuno steerer attivato."); return

        try:
            df = pd.read_csv(prompts_file, sep=';')
            if max_samples: df = df.head(max_samples)
        except: print("❌ Error reading CSV"); return

        os.makedirs(output_dir, exist_ok=True)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                pid = str(row.get('ID', row.get('id', i))).strip()
                if 'test_prompt' in row: prompt = str(row['test_prompt']).strip()
                elif 'prompt' in row: prompt = str(row['prompt']).strip()
                else: prompt = str(row.iloc[-1]).strip()
                
                # Nomi file puliti per Evaluation: {ID}_xxx.wav
                base = os.path.join(output_dir, f"{pid}")

                self.mg.generate(prompt, f"{base}_orig")
                
                for s in steerers: s.apply(alpha)
                self.mg.generate(prompt, f"{base}_pos")
                for s in steerers: s.remove()
                
                for s in steerers: s.apply(-alpha)
                self.mg.generate(prompt, f"{base}_neg")
                for s in steerers: s.remove()
                
            except Exception as e:
                print(f"Error {pid}: {e}")
                for s in steerers: s.remove()
