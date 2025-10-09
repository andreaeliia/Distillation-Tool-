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
    """Adapter per caricare e gestire dataset da CSV"""
    
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    TEXT_LENGTH_THRESHOLD = 100
    MAX_TOKEN_LENGTH = 512
    
    def __init__(self, csv_path, tokenizer_name=None, max_samples=None, 
                 imagenet_mapping_path=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.imagenet_mapping_path = imagenet_mapping_path
        
        if max_samples:
            print(f"[DATASET] Limitazione a {max_samples} campioni")
            self.df = self.df.head(max_samples)
        
        print(f"[DATASET] Caricato dataset: {self.df.shape}")
        
        self.mode = self._infer_mode()
        print(f"[DATASET] Modalità rilevata: {self.mode}")
        
        self.tokenizer = None
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self._setup_label_mapping()
    
    def _setup_label_mapping(self):
        """Configura il mapping delle label"""
        if self.imagenet_mapping_path and os.path.exists(self.imagenet_mapping_path):
            print(f"[DATASET] Caricamento mapping da: {self.imagenet_mapping_path}")
            self._load_mapping_from_file()
        else:
            print("[DATASET] Creazione mapping automatico")
            self._create_automatic_mapping()
        
        self._apply_mapping()
    
    def _load_mapping_from_file(self):
        """Carica mapping da file JSON"""
        with open(self.imagenet_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        self.label_to_idx = mapping_data.get('label_to_idx', {})
        self.idx_to_label = {int(k): v for k, v in mapping_data.get('idx_to_label', {}).items()}
        
        print(f"[DATASET] Mapping caricato: {len(self.label_to_idx)} classi")
    
    def _create_automatic_mapping(self):
        """Crea mapping automatico dalle label uniche"""
        unique_labels = self.df.iloc[:, 1].unique()
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        for idx, label in enumerate(sorted(unique_labels)):
            label_str = str(label)
            self.label_to_idx[label_str] = idx
            self.idx_to_label[idx] = label_str
        
        print(f"[DATASET] Mapping creato: {len(unique_labels)} classi")
    
    def _apply_mapping(self):
        """Applica il mapping delle label al DataFrame"""
        self.df.iloc[:, 1] = self.df.iloc[:, 1].astype(str)
        original_labels = self.df.iloc[:, 1].copy()
        mapped_labels = self.df.iloc[:, 1].map(self.label_to_idx)
        
        unmapped_mask = mapped_labels.isna()
        if unmapped_mask.any():
            unmapped_labels = original_labels[unmapped_mask].unique()
            print(f"[WARNING] Label non mappate: {unmapped_labels[:10]}")
            mapped_labels = mapped_labels.fillna(0)
        
        self.df.iloc[:, 1] = mapped_labels.astype(int)
        print(f"[DATASET] Mapping applicato. Range: {mapped_labels.min()}-{mapped_labels.max()}")
    
    def _infer_mode(self):
        """Inferisce il tipo di dataset dal contenuto"""
        first_col = self.df.iloc[:, 0].astype(str)
        
        if first_col.str.contains(r'\.(jpg|jpeg|png)$', case=False).any():
            return "image"
        if first_col.str.startswith("data:image/").any():
            return "image"
        if first_col.str.len().mean() > self.TEXT_LENGTH_THRESHOLD:
            return "text"
        
        return "tabular"
    
    def get_dataset_info(self):
        """Ritorna informazioni sul dataset"""
        return {
            'num_classes': len(self.label_to_idx),
            'class_names': list(self.label_to_idx.keys()),
            'label_mapping': self.label_to_idx,
            'task_type': self.mode,
            'num_samples': len(self.df)
        }
    
    def get_num_classes(self):
        """Ritorna il numero di classi"""
        return len(self.label_to_idx)
    
    def save_mapping(self, output_path):
        """Salva il mapping in un file JSON"""
        mapping_data = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'num_classes': len(self.label_to_idx)
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"[DATASET] Mapping salvato in: {output_path}")
    
    def print_mapping_info(self):
        """Stampa informazioni sul mapping"""
        print("[DATASET] Informazioni mapping:")
        print(f"  Numero classi: {len(self.label_to_idx)}")
        print(f"  Range indici: 0 - {len(self.label_to_idx) - 1}")
        
        sample_size = min(10, len(self.label_to_idx))
        print(f"  Esempi (primi {sample_size}):")
        for label, idx in list(self.label_to_idx.items())[:sample_size]:
            print(f"    '{label}' -> {idx}")
    
    def get_dataloader(self):
        """Ritorna il DataLoader appropriato in base al tipo di dataset"""
        loaders = {
            "image": self._get_image_loader,
            "text": self._get_text_loader,
            "tabular": self._get_tabular_loader
        }
        return loaders[self.mode]()
    
    def _get_image_loader(self):
        """Crea DataLoader per immagini"""
        transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.NORMALIZE_MEAN, std=self.NORMALIZE_STD),
        ])
        
        class ImageDataset(Dataset):
            def __init__(self, df, transform):
                self.paths = df.iloc[:, 0].tolist()
                self.labels = df.iloc[:, 1].tolist()
                self.transform = transform
            
            def __len__(self):
                return len(self.paths)
            
            def __getitem__(self, idx):
                image = self._load_image(self.paths[idx])
                if self.transform:
                    image = self.transform(image)
                return image, int(self.labels[idx])
            
            def _load_image(self, img_path):
                try:
                    if img_path.startswith("data:image"):
                        img_data = base64.b64decode(img_path.split(",")[1])
                        return Image.open(io.BytesIO(img_data)).convert("RGB")
                    
                    if os.path.exists(img_path):
                        return Image.open(img_path).convert("RGB")
                    
                    print(f"[WARNING] File non trovato: {img_path}")
                except Exception as e:
                    print(f"[ERROR] Errore caricamento: {e}")
                
                return Image.new('RGB', BaseDatasetAdapter.IMAGE_SIZE, color='gray')
        
        dataset = ImageDataset(self.df, transform)
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
    
    def _get_text_loader(self):
        """Crea DataLoader per testo"""
        if not self.tokenizer:
            raise ValueError("Tokenizer necessario per dataset testuali")
        
        class TextDataset(Dataset):
            def __init__(self, df):
                self.texts = df.iloc[:, 0].astype(str).tolist()
                self.labels = df.iloc[:, 1].tolist()
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]
        
        def collate_fn(batch):
            texts, labels = zip(*batch)
            tokenized = self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_TOKEN_LENGTH
            )
            return {**tokenized, 'labels': torch.tensor(labels)}
        
        dataset = TextDataset(self.df)
        return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    def _get_tabular_loader(self):
        """Crea DataLoader per dati tabulari"""
        class TabularDataset(Dataset):
            def __init__(self, df):
                self.x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
                self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        
        dataset = TabularDataset(self.df)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    def get_generation_loader(self):
        """Crea DataLoader per text generation"""
        if not self.tokenizer:
            raise ValueError("Tokenizer necessario per text generation")
        
        class GenerationDataset(Dataset):
            def __init__(self, df):
                self.inputs = df.iloc[:, 0].astype(str).tolist()
                self.targets = df.iloc[:, 1].astype(str).tolist()
            
            def __len__(self):
                return len(self.inputs)
            
            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]
        
        def collate_fn(batch):
            inputs, targets = zip(*batch)
            
            input_encoding = self.tokenizer(
                list(inputs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_TOKEN_LENGTH
            )
            
            target_encoding = self.tokenizer(
                list(targets),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_TOKEN_LENGTH
            )
            
            return {
                'input_ids': input_encoding['input_ids'],
                'attention_mask': input_encoding['attention_mask'],
                'target_ids': target_encoding['input_ids'],
                'target_attention_mask': target_encoding['attention_mask']
            }
        
        dataset = GenerationDataset(self.df)
        return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)


def create_imagenet_mapping_from_train_dir(train_dir, output_path):
    """Crea mapping ImageNet dalle cartelle di training"""
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory train non trovata: {train_dir}")
    
    class_dirs = sorted([d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))])
    
    label_to_idx = {class_dir: idx for idx, class_dir in enumerate(class_dirs)}
    idx_to_label = {idx: class_dir for idx, class_dir in enumerate(class_dirs)}
    
    mapping_data = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': len(class_dirs),
        'created_from': train_dir
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"[MAPPING] ImageNet creato: {len(class_dirs)} classi -> {output_path}")
    return mapping_data


def count_classes(csv_path):
    """Conta rapidamente il numero di classi nel CSV"""
    df = pd.read_csv(csv_path)
    return len(df.iloc[:, 1].unique())