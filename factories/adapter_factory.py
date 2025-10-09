import os
from adapters.dataset_adapter import BaseDatasetAdapter
from adapters.model_adapter import BaseModelAdapter


class AdapterFactory:
    """Factory per creare adapter per dataset e modelli"""
    
    @staticmethod
    def _validate_path(path, path_type="File"):
        """Valida l'esistenza di un path"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_type} non trovato: {path}")
    
    @staticmethod
    def create_dataset_adapter(dataset_path, tokenizer_name=None, max_samples=None, 
                                imagenet_mapping_path=None):
        """
        Crea un dataset adapter e ritorna adapter + informazioni dataset
        
        Args:
            dataset_path: Path al file CSV del dataset
            tokenizer_name: Nome del tokenizer per dataset testuali
            max_samples: Numero massimo di campioni da caricare
            imagenet_mapping_path: Path al file di mapping ImageNet
            
        Returns:
            tuple: (adapter, dataset_info)
        """
        AdapterFactory._validate_path(dataset_path, "Dataset")
        
        print(f"[FACTORY] Creazione dataset adapter: {dataset_path}")
        
        adapter = BaseDatasetAdapter(
            csv_path=dataset_path,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples,
            imagenet_mapping_path=imagenet_mapping_path
        )
        
        dataset_info = adapter.get_dataset_info()
        
        print(f"[FACTORY] Dataset creato:")
        print(f"  - Task type: {dataset_info['task_type']}")
        print(f"  - Shape: {adapter.df.shape}")
        print(f"  - Classi: {dataset_info['num_classes']}")
        
        return adapter, dataset_info
    
    @staticmethod
    def create_model_adapter(model_path):
        """
        Crea un model adapter
        
        Args:
            model_path: Path al modello salvato (.pt)
            
        Returns:
            BaseModelAdapter: Adapter per il modello
        """
        AdapterFactory._validate_path(model_path, "Modello")
        
        print(f"[FACTORY] Creazione model adapter: {model_path}")
        
        adapter = BaseModelAdapter(model_path)
        
        print(f"[FACTORY] Modello creato: {type(adapter.model).__name__}")
        
        return adapter
    
    @staticmethod
    def create_imagenet_dataset_adapter(dataset_path, imagenet_mapping_path, 
                                        tokenizer_name=None, max_samples=None):
        """
        Crea dataset adapter specifico per ImageNet
        
        Args:
            dataset_path: Path al file CSV del dataset
            imagenet_mapping_path: Path al file di mapping ImageNet
            tokenizer_name: Nome del tokenizer (opzionale)
            max_samples: Numero massimo di campioni
            
        Returns:
            tuple: (adapter, dataset_info)
        """
        print("[FACTORY] Creazione dataset adapter ImageNet")
        
        return AdapterFactory.create_dataset_adapter(
            dataset_path=dataset_path,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples,
            imagenet_mapping_path=imagenet_mapping_path
        )