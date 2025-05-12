import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import io
import base64

"""
Adapter class that automatically determines the type of dataset (image, text, or tabular)
from a given CSV file and returns the appropriate PyTorch DataLoader.
It uses heuristics to infer the modality of the data and applies transformations
suitable for the model input.
"""
class BaseDatasetAdapter:
    def __init__(self, csv_path, tokenizer_name="bert-base-uncased"):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.mode = self.infer_mode()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # Carica il tokenizer

    """
    Infer the type of data contained in the CSV file. If any value in the first row
    appears to be an image (data URI or file path), it returns 'image'. If the average
    string length of the row is longer than a threshold (suggesting sentence-like data),
    it returns 'text'. Otherwise, it defaults to 'tabular'. This heuristic can be
    overridden by subclassing if more robust or domain-specific logic is needed.
    """
    def infer_mode(self):
        sample = self.df.iloc[0]
        if any(sample.astype(str).str.startswith("data:image/") | sample.astype(str).str.endswith(".png") | sample.astype(str).str.endswith(".jpg")):
            return "image"
        elif sample.astype(str).str.len().mean() > 20:  # Da rivedere, per ora assumo che i testi siano più lunghi
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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        class ImageDataset(Dataset):
            def __init__(self, df):
                self.df = df

            def __getitem__(self, idx):
                val = self.df.iloc[idx, 0]
                if val.startswith("data:image"):
                    image = Image.open(io.BytesIO(base64.b64decode(val.split(",")[1])))
                else:
                    image = Image.open(val)
                return transform(image)

            def __len__(self):
                return len(self.df)

        return DataLoader(ImageDataset(self.df), batch_size=16)

    def get_text_loader(self):
        class TextDataset(Dataset):
            def __init__(self, df):
                self.texts = df.iloc[:, 0].astype(str).tolist()

            def __getitem__(self, idx):
                text = self.texts[idx]
                # Note: max_length=512 is a common default but may need adjustment
                # depending on the tokenizer or model used
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                return inputs

            def __len__(self):
                return len(self.texts)

        return DataLoader(TextDataset(self.df), batch_size=16)

    def get_tabular_loader(self):
        class TabularDataset(Dataset):
            def __init__(self, df):
                # Assumes all columns in the CSV are numerical and suitable for conversion to float tensors.
                self.data = torch.tensor(df.values, dtype=torch.float32)

            def __getitem__(self, idx):
                return self.data[idx]

            def __len__(self):
                return len(self.data)

        return DataLoader(TabularDataset(self.df), batch_size=16)
