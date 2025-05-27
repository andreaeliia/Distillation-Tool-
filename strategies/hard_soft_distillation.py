import torch
import torch.nn.functional as F
from strategies.base import DistillationStrategy
from tqdm import tqdm
from config.constants import TEMPERATURE, ALPHA

class HardSoftDistillation(DistillationStrategy):
    def __init__(self, temperature=TEMPERATURE, alpha=ALPHA, chunk_size=None):
        self.temperature = temperature
        self.alpha = alpha
        self.chunk_size = chunk_size



    def distill(self, teacher_adapter, student_adapter, dataloader, temperature=TEMPERATURE, alpha=ALPHA):
        optimizer = torch.optim.Adam(student_adapter.model.parameters(), lr=1e-4)

        for batch in tqdm(dataloader):
            inputs = {k: v.to(student_adapter.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(student_adapter.device)
            with torch.no_grad():
                teacher_logits = teacher_adapter.predict_logits(inputs)
            student_logits = student_adapter.predict_logits(inputs)

            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean"
            ) * (temperature ** 2)

            hard_loss = F.cross_entropy(student_logits, labels)

            loss = alpha * hard_loss + (1 - alpha) * soft_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()