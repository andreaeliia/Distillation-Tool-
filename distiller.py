from factories.adapter_factory import AdapterFactory
from factories.strategy_factory import StrategyFactory
from utils.save_model import save_model
import os
import torch

class DistillerBridge:
    def __init__(self, teacher_path, student_path, dataset_path,output_path, distillation_strategy,tokenizer_name = 'bert-base-uncased'):
        self.student_adapter = AdapterFactory.create_model_adapter(student_path)
        self.teacher_adapter = AdapterFactory.create_model_adapter(teacher_path)
        self.dataset_adapter = AdapterFactory.create_dataset_adapter(dataset_path, tokenizer_name)
        self.strategy = StrategyFactory.create_strategy(self.teacher_adapter, self.student_adapter,strategy_name=distillation_strategy)

        self.output_path = output_path

    def distill(self):
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
            #TODO : sistemare questa parte , messa solo per sbrigarmi
            torch.save(self.student_adapter.model, model_save_path)
            print(f"💾 Modello student salvato correttamente in: {model_save_path}")
        except Exception as e:
            print(f"❌ Errore durante il salvataggio del modello: {e}")
