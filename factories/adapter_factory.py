from adapters.model_adapter import BaseModelAdapter
from adapters.dataset_adapter import BaseDatasetAdapter

class AdapterFactory:
    @staticmethod
    def create_model_adapter(model_path):
        return BaseModelAdapter(model_path)

    @staticmethod
    def create_dataset_adapter(dataset_path, tokenizer_name):
        return BaseDatasetAdapter(dataset_path, tokenizer_name)
    
    @staticmethod 
    def create_dataset_adapter(dataset_path,imagenet_mapping_path):
        return BaseDatasetAdapter(dataset_path,imagenet_mapping_path)