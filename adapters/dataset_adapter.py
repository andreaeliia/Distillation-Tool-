import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import json
from config.constants import BATCH_SIZE
from transformers import AutoTokenizer


class BaseDatasetAdapter:
    def __init__(self, csv_path, tokenizer_name=None, max_samples=None, imagenet_mapping_path=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.imagenet_mapping_path = imagenet_mapping_path

        if max_samples:
            print(f"[INFO] Trimming dataset to max {max_samples} samples")
            self.df = self.df.head(max_samples)

        print(f"[INFO] Loaded dataset with shape: {self.df.shape}")
        self.mode = self.infer_mode()
        print(f"[INFO] Inferred mode: {self.mode}")
        self.tokenizer = None
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


        # Gestione mapping label (ImageNet o generico)
        self.setup_label_mapping()

    def setup_label_mapping(self):
        """
        Configura il mapping delle label - può essere specifico per ImageNet o generico
        """
        if self.imagenet_mapping_path and os.path.exists(self.imagenet_mapping_path):
            print(f"[INFO] Caricamento mapping ImageNet da: {self.imagenet_mapping_path}")
            self.load_imagenet_mapping()
        else:
            print(f"[INFO] Creazione mapping automatico dalle label del dataset")
            self.create_automatic_mapping()
        
        # Applica il mapping al DataFrame
        self.apply_label_mapping()

    def load_imagenet_mapping(self):
        """
        Carica il mapping ImageNet da file JSON
        """
        with open(self.imagenet_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        self.label_to_idx = mapping_data.get('label_to_idx', {})
        self.idx_to_label = mapping_data.get('idx_to_label', {})
        # Converti le chiavi numeriche di idx_to_label in int
        self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
        
        print(f"[INFO] Mapping ImageNet caricato: {len(self.label_to_idx)} classi")

    def create_automatic_mapping(self):
        """
        Crea mapping automatico dalle label uniche nel dataset
        """
        unique_labels = self.df.iloc[:, 1].unique()
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        for idx, label in enumerate(sorted(unique_labels)):
            self.label_to_idx[str(label)] = idx  # Assicura che sia stringa
            self.idx_to_label[idx] = str(label)
        
        print(f"[INFO] Mapping automatico creato: {len(unique_labels)} classi")

    def apply_label_mapping(self):
        """
        Applica il mapping delle label al DataFrame
        """
        # Converte le label in stringhe per il mapping
        self.df.iloc[:, 1] = self.df.iloc[:, 1].astype(str)
        
        # Applica il mapping
        original_labels = self.df.iloc[:, 1].copy()
        mapped_labels = self.df.iloc[:, 1].map(self.label_to_idx)
        
        # Controlla se ci sono label non mappate
        unmapped_mask = mapped_labels.isna()
        if unmapped_mask.any():
            unmapped_labels = original_labels[unmapped_mask].unique()
            print(f"[WARNING] Label non mappate trovate: {unmapped_labels[:10]}...")  # Mostra prime 10
            # Assegna 0 alle label non mappate (o potresti voler gestire diversamente)
            mapped_labels = mapped_labels.fillna(0)
        
        self.df.iloc[:, 1] = mapped_labels.astype(int)
        print(f"[INFO] Label mappate con successo. Range: {mapped_labels.min()}-{mapped_labels.max()}")

    def save_mapping(self, output_path):
        """
        Salva il mapping corrente in un file JSON
        """
        mapping_data = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'num_classes': len(self.label_to_idx)
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"[INFO] Mapping salvato in: {output_path}")

    def infer_mode(self):
        first_col = self.df.iloc[:, 0].astype(str)
        if first_col.str.contains(r'\.(jpg|jpeg|png)$', case=False).any():
            return "image"
        elif first_col.str.startswith("data:image/").any():
            return "image"
        elif first_col.str.len().mean() > 100:
            return "text"
        else:
            return "tabular"


    def get_dataloader(self):
        if self.mode == "image":
            return self.get_image_loader()
        elif self.mode == "text":
            return self.get_text_loader()
        else:
            return self.get_tabular_loader()

    def get_image_loader(self):
        # Trasformazioni standard per ImageNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        class ImageDataset(Dataset):
            def __init__(self, df, transform, label_to_idx):
                self.paths = df.iloc[:, 0].tolist()
                self.labels = df.iloc[:, 1].tolist()
                self.transform = transform
                self.label_to_idx = label_to_idx

            def __getitem__(self, idx):
                img_path = self.paths[idx]
                label = self.labels[idx]
                
                try:
                    if img_path.startswith("data:image"):
                        # Gestione base64
                        image = Image.open(io.BytesIO(base64.b64decode(img_path.split(",")[1]))).convert("RGB")
                    elif os.path.exists(img_path):
                        # Gestione file path
                        image = Image.open(img_path).convert("RGB")
                    else:
                        print(f"[WARNING] File non trovato: {img_path}")
                        image = Image.new('RGB', (224, 224), color='gray')
                except Exception as e:
                    print(f"[ERROR] Errore caricamento immagine {img_path}: {e}")
                    image = Image.new('RGB', (224, 224), color='gray')
                
                if self.transform:
                    image = self.transform(image)
                
                return image, int(label)

            def __len__(self):
                return len(self.paths)

        dataset = ImageDataset(self.df, transform, self.label_to_idx)
        return DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    def get_text_loader(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for text mode.")

        class TextDataset(Dataset):
            def __init__(self, df):
                self.texts = df.iloc[:, 0].astype(str).tolist()
                self.labels = df.iloc[:, 1].tolist()


            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]

            def __len__(self):
                return len(self.texts)

        def collate_fn(batch):
            texts, labels = zip(*batch)
            tokenized = self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            return {**tokenized, 'labels': torch.tensor(labels)}

        return DataLoader(TextDataset(self.df), batch_size=BATCH_SIZE, collate_fn=collate_fn)

    def get_tabular_loader(self):
        class TabularDataset(Dataset):
            def __init__(self, df):
                self.x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
                self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

            def __len__(self):
                return len(self.x)

        return DataLoader(TabularDataset(self.df), batch_size=BATCH_SIZE, shuffle=True)

    def get_num_classes(self):
        """
        Ritorna il numero di classi
        """
        return len(self.label_to_idx)

    def print_mapping_info(self):
        """
        Stampa informazioni sul mapping delle classi
        """
        print(f"[INFO] === MAPPING INFO ===")
        print(f"Numero classi: {len(self.label_to_idx)}")
        print(f"Range indici: 0 - {len(self.label_to_idx) - 1}")
        
        # Mostra alcuni esempi di mapping
        sample_size = min(10, len(self.label_to_idx))
        sample_items = list(self.label_to_idx.items())[:sample_size]
        print(f"Esempi mapping (primi {sample_size}):")
        for label, idx in sample_items:
            print(f"  '{label}' -> {idx}")
        print(f"[INFO] ====================")


# ===== FUNZIONI DI UTILITÀ PER IMAGENET =====

def create_imagenet_mapping_from_train_dir(train_dir, output_path):
    """
    Crea il mapping ImageNet dalle cartelle di training
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory train non trovata: {train_dir}")
    
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_dirs.sort()  # Ordinamento consistente
    
    label_to_idx = {}
    idx_to_label = {}
    
    for idx, class_dir in enumerate(class_dirs):
        label_to_idx[class_dir] = idx
        idx_to_label[idx] = class_dir
    
    mapping_data = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': len(class_dirs),
        'created_from': train_dir
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"[INFO] Mapping ImageNet creato: {len(class_dirs)} classi -> {output_path}")
    return mapping_data