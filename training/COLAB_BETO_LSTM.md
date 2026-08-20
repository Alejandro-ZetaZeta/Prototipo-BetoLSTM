# Google Colab: BETO + BiLSTM

Recommended: open `training/colab_train_beto_lstm.ipynb` in Colab and execute cells with `Shift+Enter`. It mounts the existing Drive project, translates the XLSX, trains, and saves the new artifact to Drive.

Steps below document same notebook flow manually.

Run notebook cells in order. GPU: **Runtime > Change runtime type > T4 GPU**.

## 1. Upload project

```python
from google.colab import files
uploaded = files.upload()  # upload Protoype-Elbeto.zip
!unzip -q Protoype-Elbeto.zip
%cd Protoype-Elbeto
```

If project is stored in Google Drive, mount Drive instead and set `ROOT` to the project path.

## 2. Install dependencies

```python
!pip install -q -r training/requirements-etl.txt
!pip install -q accelerate
```

## 3. Prepare translated data

The XLSX requirements are English. Do **not** pass `--skip-translate`.
The command below translates English text to Spanish, maps `FR -> 0` and `NFR -> 1`, removes duplicate texts, and creates stratified partitions.

```python
!python training/etl_prepare.py \
  --input training/Dataset6000Req.xlsx \
  --additional-input training/dataset_raw.csv \
  --output-dir training \
  --dataset-name combined
```

Translation uses a persistent cache at `training/.translate_cache.json`. Save this cache to Drive if the Colab session may restart.

Check output before training:

```python
import pandas as pd

train = pd.read_csv("training/combined_train.csv")
print(train.head())
print(train["label"].value_counts())
```

Requirements should be Spanish. Labels must contain both `0` and `1`.

## 4. Train

The script initializes the encoder from the current BETO artifact, freezes embeddings and the first 9 BETO layers, and trains the last 3 layers plus BiLSTM classifier.

```python
!python training/train_beto_lstm.py \
  --data-prefix combined \
  --device cuda \
  --epochs 6 \
  --batch-size 16
```

Output:

```text
backend/models/beto_lstm_rf_rnf/
```

## 5. Save artifact

```python
!zip -qr beto_lstm_rf_rnf.zip backend/models/beto_lstm_rf_rnf
from google.colab import files
files.download("beto_lstm_rf_rnf.zip")
```

Copy the extracted directory into the local project's `backend/models/` directory. The existing BETO, Random Forest, and Naive Bayes artifacts remain unchanged.
