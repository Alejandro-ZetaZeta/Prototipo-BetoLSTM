# Project Context: Requirements Classifier (RF vs NNF) with BETO

## 1. Overview (All instructions will be in English, but the model must be trained in and understand Spanish)

The objective of this project is to build a functional prototype to automatically classify software specifications in Spanish into two categories using Artificial Intelligence:

- **RF (Functional Requirements):** System actions, capabilities, and behavior (Label: `0`).

- **RNF (Non-Functional Requirements):** Performance, security, usability, and quality constraints (Label: `1`).

The selected base model is **BETO** (`dccuchile/bert-base-spanish-wwm-cased`). In this first phase, BETO will be trained exclusively using fine-tuning (without additional BiLSTM layers).

---

## 2. Repository Structure (Monorepo)

```text
├── training/
│ ├── dataset_raw.csv # Original dataset
│ ├── dataset_clean.csv # Processed dataset (text, label)
│ ├── etl_prepare.py # Cleaning, translation, and partitioning script
│ └── colab_train_beto.ipynb # Fine-tuning notebook for Google Colab
├── backend/
│ ├── app/
│ │ ├── main.py # FastAPI entrypoint
│ │ ├── model.py # Load BETO and pipeline Inference
│ │ └── schemas.py # Pydantic schemas (Request / Response)
│ ├── models/ # Exported artifacts (weights, vocabulary, configuration)
│ ├── requirements.txt # Python dependencies (FastAPI, PyTorch, etc.)
│ └── .env.example
├── frontend/ # Web client (Astro + Tailwind)
└── AGENTS.md
```

## 3. Active Tasks (Phase 1: Preparation and Training)

Task 1.1: Data Pipeline and ETL (training/etl_prepare.py)
Load requirements datasets (e.g., PROMISE NFR Dataset).

Clean the text: normalize spacing, remove unnecessary special characters, and convert to standard format.

Translate or verify sentences in Spanish.

Map classes to binary format: 0 for RF and 1 for RNF.

Split the dataset into train (70%), val (15%), and test (15%) using stratification (stratify=y).

Export partitions to CSV files ready for training.

Task 1.2: Training Script/Notebook (training/colab_train_beto.ipynb)
Load the model and tokenizer: dccuchile/bert-base-spanish-wwm-cased using Hugging Face transformers.

Configure tokenization with truncation and padding (max_length=128).

Layer Freezing: Freeze the first 9 layers of the BETO encoder and allow training only on the last 3 layers and the classifier head (BertForSequenceClassification).

Base hyperparameters:

Optimizer: AdamW

Learning Rate: 2e-5 to 3e-5

Batch size: 16 or 32

Epoches: 5 to 10 with continuous evaluation

Metrics to report: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

Save the final artifact with `model.save_pretrained("./models/beto_rf_rnf")` and `tokenizer.save_pretrained("./models/beto_rf_rnf")`.

4. Future Tasks (Phase 2: Backend and Local Inference)
   Task 2.1: Inference Microservice (backend/)
   Build a REST API in FastAPI that loads the trained weights into the local CPU.

Endpoint POST /api/predict:

Input:
{"text": "El sistema debe encriptar las contraseñas con SHA-256"}

Output:
{"label": "RNF", "confidence": 0.96, "execution_time_ms": 24.5}

Endpoint POST /api/predict/batch: Allows sending lists or files of requirements.

5. Future Tasks (Phase 3: Web Interface and Persistence)
   Task 3.1: Frontend
   Create a minimalist and interactive web interface.

Quick input form for individual sentences.

Table for displaying batch predictions.

Cards with color-coded visual indicators (Blue = RF, Green/Orange = RNF) and a confidence bar.

Task 3.2: Persistence (Optional)
Store the query and classification history in a relational database (PostgreSQL / TiDB).

6. Development Conventions
   Python: Version >= 3.10, strict use of type hints and docstrings.

Tensor handling: Ensure inference compatibility on both GPUs (CUDA) and CPUs (CPU).

Dependency control: Keep requirements.txt files clean and free of redundant libraries.
