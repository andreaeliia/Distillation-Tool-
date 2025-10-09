import pandas as pd
from dataset_by_tasking.task_type import TaskType
from typing import Dict, Any


class TaskDetector:
    """Rilevamento automatico del tipo di task dal dataset"""
    
    # Colonne comuni per input e target
    INPUT_KEYWORDS = ['text', 'sentence', 'review', 'comment', 'image_path', 
                      'image', 'input', 'question', 'context']
    TARGET_KEYWORDS = ['label', 'target', 'class', 'category', 'sentiment', 
                       'score', 'answer']
    
    @staticmethod
    def detect_task_type(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Rileva il tipo di task analizzando le colonne del DataFrame
        
        Returns:
            dict: Informazioni sul task rilevato
        """
        columns = df.columns.tolist()
        print(f"[TASK DETECTOR] Analisi colonne: {columns}")
        
        # Identifica colonne input e target
        input_cols = TaskDetector._identify_input_columns(columns)
        target_cols = TaskDetector._identify_target_columns(columns, input_cols)
        
        print(f"[TASK DETECTOR] Input: {input_cols}")
        print(f"[TASK DETECTOR] Target: {target_cols}")
        
        # Determina tipo di task e dati
        task_type = TaskDetector._determine_task_type(df, input_cols, target_cols)
        data_type = TaskDetector._determine_data_type(df, input_cols)
        
        # Analizza target per classificazione
        num_classes = None
        is_multiclass = False
        
        if target_cols and 'classification' in task_type.value:
            target_col = target_cols[0]
            num_classes = df[target_col].nunique()
            is_multiclass = num_classes > 2
            print(f"[TASK DETECTOR] Numero classi: {num_classes}")
        
        print(f"[TASK DETECTOR] Task rilevato: {task_type.value}")
        
        return {
            'task_type': task_type,
            'input_columns': input_cols,
            'target_columns': target_cols,
            'num_classes': num_classes,
            'is_multiclass': is_multiclass,
            'data_type': data_type
        }
    
    @staticmethod
    def _identify_input_columns(columns: list) -> list:
        """Identifica le colonne di input"""
        input_cols = [col for col in columns 
                     if col.lower() in TaskDetector.INPUT_KEYWORDS]
        
        if not input_cols:
            # Usa la prima colonna escludendo quelle che sembrano target
            potential = [col for col in columns 
                        if col.lower() not in TaskDetector.TARGET_KEYWORDS]
            input_cols = [potential[0]] if potential else [columns[0]]
        
        return input_cols
    
    @staticmethod
    def _identify_target_columns(columns: list, input_cols: list) -> list:
        """Identifica le colonne target"""
        target_cols = [col for col in columns 
                      if col.lower() in TaskDetector.TARGET_KEYWORDS]
        
        if not target_cols:
            # Usa l'ultima colonna escludendo quelle di input
            potential = [col for col in columns if col not in input_cols]
            target_cols = [potential[-1]] if potential else [columns[-1]]
        
        return target_cols
    
    @staticmethod
    def _determine_task_type(df: pd.DataFrame, input_cols: list, 
                            target_cols: list) -> TaskType:
        """Determina il tipo di task"""
        if not input_cols:
            return TaskType.TEXT_CLASSIFICATION
        
        input_col = input_cols[0]
        input_name = input_col.lower()
        
        # Controllo basato sul nome della colonna
        if 'image' in input_name or 'path' in input_name:
            return TaskType.IMAGE_CLASSIFICATION
        
        if 'question' in input_name or 'context' in input_name:
            return TaskType.QUESTION_ANSWERING
        
        if any(kw in input_name for kw in ['text', 'sentence', 'review', 'comment']):
            return TaskDetector._classify_text_task(df, target_cols)
        
        # Controllo basato sul contenuto
        return TaskDetector._classify_by_content(df, input_col, target_cols)
    
    @staticmethod
    def _classify_text_task(df: pd.DataFrame, target_cols: list) -> TaskType:
        """Classifica task testuali"""
        if not target_cols:
            return TaskType.TEXT_GENERATION
        
        target_col = target_cols[0]
        unique_values = df[target_col].nunique()
        
        # Se ha poche classi discrete, è classificazione
        is_discrete = df[target_col].dtype in ['object', 'category']
        if is_discrete or unique_values <= 50:
            return TaskType.TEXT_CLASSIFICATION
        
        return TaskType.TEXT_GENERATION
    
    @staticmethod
    def _classify_by_content(df: pd.DataFrame, input_col: str, 
                            target_cols: list) -> TaskType:
        """Classifica task analizzando il contenuto"""
        sample = df[input_col].dropna().iloc[:10].astype(str)
        
        # Controlla se sono path di immagini
        is_image_path = sample.str.contains(
            r'\.(jpg|jpeg|png|bmp|tiff?)$', 
            case=False, 
            regex=True
        ).any()
        
        if is_image_path:
            return TaskType.IMAGE_CLASSIFICATION
        
        # Controlla lunghezza media per distinguere testo da dati tabulari
        avg_length = sample.str.len().mean()
        
        if avg_length > 20:
            return (TaskType.TEXT_CLASSIFICATION if target_cols 
                   else TaskType.TEXT_GENERATION)
        
        return TaskType.TABULAR_CLASSIFICATION
    
    @staticmethod
    def _determine_data_type(df: pd.DataFrame, input_cols: list) -> str:
        """Determina il tipo di dati di input"""
        if not input_cols:
            return 'unknown'
        
        input_col = input_cols[0]
        input_name = input_col.lower()
        
        # Controllo basato sul nome
        if 'image' in input_name or 'path' in input_name:
            return 'image'
        
        if any(kw in input_name for kw in ['text', 'sentence', 'review', 
                                           'comment', 'question']):
            return 'text'
        
        # Controllo basato sul contenuto
        sample = df[input_col].dropna().iloc[:5].astype(str)
        avg_length = sample.str.len().mean()
        
        if avg_length > 20:
            return 'text'
        
        is_image = sample.str.contains(
            r'\.(jpg|jpeg|png|bmp)$', 
            case=False, 
            regex=True
        ).any()
        
        return 'image' if is_image else 'numerical'