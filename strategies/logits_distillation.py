import torch
import torch.nn.functional as F
from strategies.base import DistillationStrategy
from tqdm import tqdm
from config.constants import TEMPERATURE

class LogitsDistillation(DistillationStrategy):
    def distill(self, teacher_adapter, student_adapter, dataloader):
        optimizer = torch.optim.Adam(student_adapter.model.parameters(), lr=1e-4)

        for batch in tqdm(dataloader, desc="Logits Distillation"):
            if isinstance(batch, list):
                batch = batch[0]  # E.g., lista di stringhe per NLP
            if isinstance(batch, str):
                raise ValueError("Test batches must be pre-tokenized in the DataLoader")

            with torch.no_grad():
                teacher_logits = teacher_adapter.predict_logits(batch)
            student_logits = student_adapter.predict_logits(batch)

            loss = F.kl_div(
                F.log_softmax(student_logits / TEMPERATURE, dim=1),
                F.softmax(teacher_logits / TEMPERATURE, dim=1),
                reduction="batchmean"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
