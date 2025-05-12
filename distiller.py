from factories.adapter_factory import AdapterFactory
from factories.strategy_factory import StrategyFactory

class DistillerBridge:
    def __init__(self, teacher_path, student_path, dataset_path):
        self.student_adapter = AdapterFactory.create_model_adapter(student_path)
        self.teacher_adapter = AdapterFactory.create_model_adapter(teacher_path)
        self.dataset_adapter = AdapterFactory.create_dataset_adapter(dataset_path)
        self.strategy = StrategyFactory.create_strategy(self.teacher_adapter, self.student_adapter)

    def distill(self):
        data_loader = self.dataset_adapter.get_dataloader()
        self.strategy.distill(self.teacher_adapter, self.student_adapter, data_loader)
        print('FATTO')

        #TODO salva modello