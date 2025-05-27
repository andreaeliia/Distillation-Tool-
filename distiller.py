from factories.adapter_factory import AdapterFactory
from factories.strategy_factory import StrategyFactory
from utils.save_model import save_model
import os
import torch
from config import TEMPERATURE, ALPHA

class DistillerBridge:
    def __init__(self, teacher_path, student_path, dataset_path, output_path, 
                 distillation_strategy, tokenizer_name=None, config=None):
        
        print("[BRIDGE] === INIZIALIZZAZIONE DISTILLER BRIDGE ===")
        
        # 1. MODIFICATO: Prima analizza il dataset per ottenere le info
        print("[BRIDGE] Analizzando dataset...")
        self.dataset_adapter, self.dataset_info = AdapterFactory.create_dataset_adapter(
            dataset_path, tokenizer_name
        )
        
        # 2. NUOVO: Estrai il numero di classi automaticamente
        self.num_classes = self.dataset_info['num_classes']
        print(f"[BRIDGE] 🎯 Numero classi rilevate: {self.num_classes}")
        
        # 3. Crea model adapter (rimane uguale)
        self.student_adapter = AdapterFactory.create_model_adapter(student_path)
        self.teacher_adapter = AdapterFactory.create_model_adapter(teacher_path)
        
        # 4. NUOVO: Crea o aggiorna il config con il numero di classi
        if config is None:
            # Config di default se non fornito
            self.config = {
                'temperature':TEMPERATURE ,
                'alpha': ALPHA,
                'num_classes': self.num_classes  # ← NUMERO CLASSI AUTOMATICO
            }
            print("[BRIDGE] Config di default creato")
        else:
            # Usa il config fornito ma aggiorna num_classes
            self.config = config.copy()
            self.config['num_classes'] = self.num_classes  # ← SOVRASCRIVE con valore automatico
            print("[BRIDGE] Config fornito aggiornato con numero classi automatico")
        
        print(f"[BRIDGE] Config finale: {self.config}")
        
        # 5. MODIFICATO: Crea task adapter passando il config aggiornato
        print("[BRIDGE] Creando task adapter...")
        self.task_adapter = AdapterFactory.create_task_adapter(
            dataset_path, 
            self.config,  # ← Config con num_classes automatico
            self.teacher_adapter.model, 
            self.student_adapter.model
        )
        
        # 6. Crea strategia (rimane uguale)
        self.strategy = StrategyFactory.create_strategy(
            self.teacher_adapter, 
            self.student_adapter,
            strategy_name=distillation_strategy
        )

        self.output_path = output_path
        
        print(f"[BRIDGE] === CONFIGURAZIONE COMPLETATA ===")
        print(f"  - Task: {self.dataset_info['task_type']}")
        print(f"  - Classi: {self.num_classes}")
        print(f"  - Campioni: {self.dataset_info['num_samples']}")
        print(f"  - Strategia: {distillation_strategy}")

    def get_num_classes(self):
        """Getter per il numero di classi"""
        return self.num_classes
    
    def get_config(self):
        """Getter per il config completo"""
        return self.config

    def distill(self):
        """Esegue la distillazione (rimane uguale al tuo codice originale)"""
        data_loader = self.dataset_adapter.get_dataloader()

        print("Inizio della distillazione...")
        self.strategy.distill(self.teacher_adapter, self.student_adapter, data_loader)

        print(f"✅ Student model after distillation: {self.student_adapter.model}")

        if self.student_adapter.model is None:
            print("❌ Attenzione: il modello student è None!")
        else:
            total_params = sum(p.numel() for p in self.student_adapter.model.parameters())
            print(f"✅ Il modello student ha {total_params} parametri.")

        # 📁 Salvataggio sicuro del modello
        model_save_path = os.path.join(self.output_path, "student.pt")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        try:
            torch.save(self.student_adapter.model, model_save_path)
            print(f"💾 Modello student salvato correttamente in: {model_save_path}")
        except Exception as e:
            print(f"❌ Errore durante il salvataggio del modello: {e}")