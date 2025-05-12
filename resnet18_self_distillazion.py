import os
import torch
import base64
import pandas as pd
from io import BytesIO
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage
from distiller import DistillerBridge

def save_cifar10_as_csv(csv_path, split="train"):
    """
    Salva CIFAR-10 in un CSV con immagini codificate in base64 PNG.
    Ogni riga contiene: image (base64), label (int)
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return

    dataset = CIFAR10(root="./data", train=(split == "train"), download=True)
    records = []

    for img_tensor, label in dataset:
        # Non è necessario ToPILImage, img_tensor è già un'immagine PIL
        buf = BytesIO()
        img_tensor.save(buf, format="PNG")  # Salviamo direttamente l'immagine come PNG
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        records.append({
            "image": f"data:image/png;base64,{img_base64}",
            "label": label
        })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Salvato dataset CIFAR10 in {csv_path}")



if __name__ == "__main__":
    dataset_relative_path = './datasets/CIFAR10/train.csv'
    models_relative_path = './models/pretrained/student.pt'

    # 1. Salva il dataset CIFAR-10 in CSV se non già presente
    save_cifar10_as_csv(dataset_relative_path)

    # 2. Salva ResNet18 come modello student se non esiste
    os.makedirs(os.path.dirname(models_relative_path), exist_ok=True)
    if not os.path.exists(models_relative_path):
        print("Salvataggio modello student (ResNet18)...")
        student = resnet18(num_classes=10)
        torch.save(student, models_relative_path)
        print("Modello salvato in student.pt")

    bridge = DistillerBridge(
        teacher_path=models_relative_path,
        student_path=models_relative_path,
        dataset_path=dataset_relative_path
    )

    bridge.distill()
