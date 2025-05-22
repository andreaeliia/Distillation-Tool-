import torch
import torch.nn.functional as F
from strategies.base import DistillationStrategy
from tqdm import tqdm
from config.constants import TEMPERATURE, ALPHA, CHUNK_SIZE, LEARNING_RATE

# 🔁 Configurazioni switchabili
USE_ADAM = True       # True per Adam, False per SGD
USE_EPOCHS = True     # True per usare epoche, False per ciclo unico
NUM_EPOCHS = 3     # Numero di epoche se USE_EPOCHS è True

class ChunkHardSoftDistillation(DistillationStrategy):
    def __init__(self, temperature=TEMPERATURE, alpha=ALPHA, chunk_size=CHUNK_SIZE):
        self.temperature = temperature
        self.alpha = alpha
        self.chunk_size = chunk_size

    def distill(self, teacher_adapter, student_adapter, dataloader):
        if USE_ADAM:
            print("[⚙️] Ottimizzatore: Adam")
            optimizer = torch.optim.Adam(student_adapter.model.parameters(), lr=LEARNING_RATE)
        else:
            print("[⚙️] Ottimizzatore: SGD")
            optimizer = torch.optim.SGD(student_adapter.model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

        if USE_EPOCHS:
            for epoch in range(NUM_EPOCHS):
                print(f"\n[🔁] Epoch {epoch + 1}/{NUM_EPOCHS}")
                self._run_epoch(teacher_adapter, student_adapter, dataloader, optimizer, epoch)
        else:
            print(f"[🔥] Inizio distillazione singolo ciclo (no epoche)")
            self._run_epoch(teacher_adapter, student_adapter, dataloader, optimizer, epoch=None)

    def _run_epoch(self, teacher_adapter, student_adapter, dataloader, optimizer, epoch=None):
        chunk_counter = 0
        chunk_loss_accum = 0.0
        total_loss_accum = 0.0
        total_steps = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}" if epoch is not None else "Distillazione"):
            try:
                # Estrazione immagini e labels
                if isinstance(batch, dict):
                    if 'images' in batch:
                        images = batch['images']
                        labels = batch['labels']
                    elif 'image' in batch:
                        images = batch['image']
                        labels = batch.get('label', batch.get('labels'))
                    elif 'data' in batch:
                        images = batch['data']home/delta-core/Scaricati/
                        labels = batch.get('target', batch.get('labels'))
                    else:
                        keys = list(batch.keys())
                        print(f"[DEBUG] Chiavi batch sconosciute: {keys}")
                        if len(keys) >= 2:
                            images = batch[keys[0]]
                            labels = batch[keys[1]]
                        else:
                            raise ValueError("Non riesco a identificare images e labels nel dizionario")

                elif isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        images, labels = batch
                    else:
                        print(f"[⚠️] Batch tuple/list lunghezza imprevista: {len(batch)}")
                        continue
                else:
                    print(f"[⚠️] Tipo batch imprevisto: {type(batch)}")
                    continue

                if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    print(f"[⚠️] Images o labels non sono tensori: {type(images)}, {type(labels)}")
                    continue

                if torch.isnan(images).any() or torch.isnan(labels).any():
                    print("[⚠️] Batch contiene NaN nei dati. Skipping.")
                    continue

                images = images.to(student_adapter.device)
                labels = labels.to(student_adapter.device)

                with torch.no_grad():
                    teacher_logits = teacher_adapter.predict_logits(images)

                student_logits = student_adapter.predict_logits(images)

                if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                    print("[❌] Student logits contiene NaN o inf!")
                    continue
                if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
                    print("[❌] Teacher logits contiene NaN o inf!")
                    continue

                if labels.min() < 0 or labels.max() >= student_logits.size(1):
                    print(f"[❌] Valori labels fuori range! labels: [{labels.min().item()}, {labels.max().item()}], classi: {student_logits.size(1)}")
                    continue

                # Soft loss
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / self.temperature, dim=1),
                    F.softmax(teacher_logits / self.temperature, dim=1),
                    reduction="batchmean"
                ) * (self.temperature ** 2)

                # Hard loss
                hard_loss = F.cross_entropy(student_logits, labels)

                if torch.isnan(soft_loss).any() or torch.isnan(hard_loss).any():
                    print(f"[❌] NaN nella soft o hard loss! Soft: {soft_loss.item()}, Hard: {hard_loss.item()}")
                    continue

                loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_adapter.model.parameters(), max_norm=5.0)
                optimizer.step()

                # Accumula
                chunk_loss_accum += loss.item()
                total_loss_accum += loss.item()
                chunk_counter += 1
                total_steps += 1

                if self.chunk_size and chunk_counter >= self.chunk_size:
                    avg_chunk_loss = chunk_loss_accum / chunk_counter
                    print(f"🔹 [Chunk completato] Media loss: {avg_chunk_loss:.4f} (step: {total_steps})")
                    chunk_loss_accum = 0.0
                    chunk_counter = 0

            except Exception as e:
                print(f"[💥] Errore nel processare il batch: {e}")
                if isinstance(batch, dict):
                    print(f"Chiavi disponibili: {list(batch.keys())}")
                elif isinstance(batch, (tuple, list)):
                    print(f"Lunghezza batch: {len(batch)}, tipi: {[type(item) for item in batch]}")
                continue

        if total_steps > 0:
            avg = total_loss_accum / total_steps
            print(f"[✅] Epoch {epoch+1 if epoch is not None else 'finale'} completata - Loss media: {avg:.4f}")
        else:
            print("[❌] Nessun batch processato con successo")
