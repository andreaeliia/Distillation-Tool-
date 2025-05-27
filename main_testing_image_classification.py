import os
import torch
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from utils.save_model import save_model
from utils.TaskDir import TaskDir
from utils.directory import ProjectStructure
from distiller import DistillerBridge
from PIL import Image
from adapters.dataset_adapter import count_classes

def save_cifar10_as_csv(csv_path, split="train"):
    """
    Salva il dataset CIFAR-10 in un CSV con due colonne: 'image_path', 'label'.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"{csv_path} già esistente, skip.")
        return

    dataset = load_dataset("cifar10", split=split)
    
    # Salva immagini e crea CSV
    images_dir = os.path.join(os.path.dirname(csv_path), "images")
    os.makedirs(images_dir, exist_ok=True)
    
    image_paths = []
    labels = []
    
    for i, sample in enumerate(dataset):
        image_path = os.path.join(images_dir, f"image_{i}.png")
        sample["img"].save(image_path)
        image_paths.append(image_path)
        labels.append(sample["label"])
    
    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })
    df.to_csv(csv_path, index=False)
    print(f"Salvato CIFAR-10 in {csv_path}")

if __name__ == "__main__":
    # Percorsi
    dataset_relative_path = './datasets/CIFAR10/train.csv'
    models_relative_path = './models/pretrained/vit-base.pt'
    model_name = "google/vit-base-patch16-224"

    # 1. Salva il dataset CIFAR-10
    save_cifar10_as_csv(dataset_relative_path)

    # 2. Salva il modello teacher (e student) da Hugging Face
    os.makedirs(os.path.dirname(models_relative_path), exist_ok=True)

    model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=count_classes(dataset_relative_path),ignore_mismatched_sizes=True )
    processor = AutoImageProcessor.from_pretrained(model_name)

    if not os.path.exists(models_relative_path):
        print("Salvataggio modello Hugging Face ViT...")
        save_model(model, models_relative_path)

    # Test tokenizzazione (processing) immagine
    print("Test processing immagine...")
    test_image = Image.new('RGB', (224, 224), color='red')  # Immagine di test
    inputs = processor(test_image, return_tensors="pt")
    print(f"Shape input processato: {inputs['pixel_values'].shape}")

    # 3. Percorso di output per la distillazione
    project = ProjectStructure()
    output_model_path = project.create_distillation_folder("vitbase", "cifar10")

    # 4. Esegui distillazione con strategia specificata
    bridge = DistillerBridge(
        teacher_path=models_relative_path,
        student_path=models_relative_path,  # Stesso modello per ora
        dataset_path=dataset_relative_path,
        output_path=output_model_path,
        distillation_strategy="hard_soft",  # Specifica la strategia
        #tokenizer_name=model_name  # In questo caso è il processor
    )

    bridge.distill()