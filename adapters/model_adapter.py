import torch

class BaseModelAdapter:
    def __init__(self, model_path):
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()

    def predict_logits(self, batch):
        return self.model(batch)