from abc import ABC,abstractmethod
import torch
from torch.utils.data import DataLoader
from typing import Dict,Any


class BaseTask(ABC):
    """Interfaccia base per tutte le task di distillazione"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_type = None
        
    @abstractmethod
    def prepare_dataset(self, dataset_adapter) -> DataLoader:
        """Prepara il dataset specifico per questa task"""
        pass
    
    @abstractmethod
    def get_teacher_model(self) -> torch.nn.Module:
        """Ritorna il modello teacher per questa task"""
        pass
    
    @abstractmethod
    def get_student_model(self) -> torch.nn.Module:
        """Ritorna il modello student per questa task"""
        pass
    '''
    @abstractmethod
    def compute_distillation_loss(self, teacher_outputs, student_outputs, targets) -> torch.Tensor:
        """Calcola la loss di distillazione specifica per questa task"""
        pass
    '''
    @abstractmethod
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Valuta il modello su questa task"""
        pass