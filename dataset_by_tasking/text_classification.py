from typing import Dict,Any
from dataset_by_tasking.base_task import BaseTask
from dataset_by_tasking.task_type import TaskType
import torch




class TextClassificationTask(BaseTask):
    def __init__(self, config: Dict[str, Any],teacher_model:torch.nn.Module,student_model: torch.nn.Module):
        super().__init__(config)
        self.task_type = TaskType.TEXT_CLASSIFICATION
        self.num_classes = config.get('num_classes', 2)  # Fallback a 2 se non c'è
        self._teacher_model = teacher_model
        self._student_model = student_model
        
    def prepare_dataset(self, dataset_adapter):
        if dataset_adapter.mode != "text":
            raise ValueError("Dataset must be in text mode for TextClassificationTask")
        return dataset_adapter.get_text_loader()
    
    def get_teacher_model(self) -> torch.nn.Module:
        return self._teacher_model

    def get_student_model(self) -> torch.nn.Module:
        return self._student_model
    """
    def compute_distillation_loss(self, teacher_outputs, student_outputs, targets):
        import torch.nn.functional as F
        
        temperature = self.config.get('temperature', 3.0)
        alpha = self.config.get('alpha', 0.8)
        
        teacher_logits = teacher_outputs.logits
        student_logits = student_outputs.logits
        
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
        
        distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
        student_loss = F.cross_entropy(student_logits, targets)
        
        return alpha * distillation_loss + (1 - alpha) * student_loss
    """
    def evaluate(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        return {'accuracy': correct / total}
