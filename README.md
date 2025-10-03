# Knowledge Distillation Framework

Un framework per fare distillazione di conoscenza tra modelli neurali. Supporta diverse architetture (ResNet, BERT, ViT) e diverse task (classificazione di immagini, testo, dati tabulari).

## Installazione

Clona la repository e installa le dipendenze:

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Il file requirements.txt contiene tutto il necessario, incluso PyTorch, Transformers, e le librerie per il tracking energetico.

## Come funziona

Il framework usa un sistema di adapter che gestisce automaticamente:
- Il caricamento dei dataset (da CSV)
- Il caricamento dei modelli (da file .pt)
- La scelta della strategia di distillazione
- Il salvataggio dei risultati

Basta specificare i percorsi e lui fa il resto.

## Uso base

### Distillazione su testo (es. BERT)

```python
from distiller import DistillerBridge

bridge = DistillerBridge(
    teacher_path='./models/pretrained/bert-base.pt',
    student_path='./models/pretrained/bert-base.pt',
    dataset_path='./datasets/SST2/train.csv',
    output_path='./saved_models/distillation/bert_exp1',
    distillation_strategy="hard_soft",
    tokenizer_name="bert-base-uncased"
)

bridge.distill()
```

### Distillazione su immagini (es. ResNet)

```python
bridge = DistillerBridge(
    teacher_path='./models/pretrained/resnet152.pt',
    student_path='./models/pretrained/resnet18.pt',
    dataset_path='./datasets/CIFAR10/train.csv',
    output_path='./saved_models/distillation/resnet_exp1',
    distillation_strategy="chunked"
)

bridge.distill()
```

## Formato del dataset

I dataset devono essere in formato CSV con almeno due colonne:

**Per immagini:**
- Prima colonna: path dell'immagine o immagine in base64
- Seconda colonna: label (nome classe o indice)

**Per testo:**
- Prima colonna: testo
- Seconda colonna: label

**Per dati tabulari:**
- Ultime colonne: features
- Ultima colonna: label

Il framework rileva automaticamente il tipo di task guardando il dataset.

## Strategie di distillazione

Puoi scegliere tra:

- **logits**: Distillazione classica con KL divergence sui logits
- **hard_soft**: Combinazione di cross-entropy (hard) e KL divergence (soft)
- **chunked**: Come hard_soft ma processa il dataset a blocchi, utile per dataset grandi tipo ImageNet

Esempio:
```python
distillation_strategy="hard_soft"  # o "logits" o "chunked"
```

## Configurazione

Puoi modificare i parametri nel file `config/constants.py`:

```python
TEMPERATURE = 2.0      # temperatura per la distillazione
ALPHA = 0.9           # peso tra hard e soft loss (1 = solo hard, 0 = solo soft)
BATCH_SIZE = 32       # dimensione batch
LEARNING_RATE = 1e-4  # learning rate
CHUNK_SIZE = 500      # dimensione chunk per strategia chunked
```

## Struttura delle directory

Il framework crea automaticamente questa struttura:

```
saved_models/
├── pretrain/
└── distillation/
    └── {nome_modello}_{etichetta}/
        ├── teacher/
        ├── student/
        └── logs_{data}/
```

## Script di esempio

La repository include diversi script pronti all'uso:

- `main_testing.py`: Distillazione BERT su SST-2
- `main_testing_image_classification.py`: Distillazione ViT su CIFAR-10
- `resnet_x_knowledge_distillation.py`: Distillazione tra ResNet di dimensioni diverse
- `resnet_knowledge_distillastion_imagenet.py`: Distillazione su ImageNet

Basta lanciare:
```bash
python main_testing.py
```

## Tracking energia e performance

Il framework traccia automaticamente:
- Consumo energetico (CPU, GPU, RAM)
- FLOPs e MACs
- Metriche di performance (accuracy, precision, recall, F1)

I log vengono salvati in `./logs/`.

## Bug comuni e soluzioni

### "Label non mappate trovate"
Succede quando le classi nel CSV non corrispondono al mapping del modello. Soluzione: controlla che le label nel CSV siano corrette o usa `imagenet_mapping_path` per fornire un mapping custom.

### "File non trovato" durante caricamento immagini
I path nel CSV devono essere assoluti o relativi alla directory di lavoro. Puoi anche usare immagini in base64 invece dei path.

### Out of memory durante distillazione
Riduci `BATCH_SIZE` in `config/constants.py`. Per dataset molto grandi usa la strategia "chunked".

### Model mismatch tra teacher e student
Assicurati che teacher e student abbiano lo stesso numero di classi in output. Il framework prova ad aggiustare automaticamente, ma in casi complessi devi modificare l'ultima layer manualmente.

### NaN nella loss
Di solito succede con learning rate troppo alto o temperature troppo bassa. Prova a ridurre il learning rate o aumentare la temperatura.

## Note su ImageNet

Per ImageNet serve un setup particolare perché ha 1000 classi. Lo script `resnet_knowledge_distillastion_imagenet.py` mostra come:
1. Generare il mapping delle classi dalle cartelle di training
2. Creare il CSV con i path corretti
3. Usare la strategia "chunked" per gestire il dataset grande

Il mapping viene salvato automaticamente e riusato.
