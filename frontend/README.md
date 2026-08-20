# Elbeto frontend

## Run locally

Start FastAPI from project root:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Start Astro in a second terminal:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:4321`.

The page defaults to `http://127.0.0.1:8000`. Override it with:

```powershell
$env:PUBLIC_API_URL = "http://127.0.0.1:8000"
npm run dev
```

Benchmark mode compares BETO, Random Forest, and Naive Bayes. Comparative batch mode remains available for one requirement per line. Frontend does not persist submitted text.

Train baseline artifacts from project root:

```powershell
python training/train_baselines.py
```

The benchmark endpoints are `POST /api/predict/benchmark` and `POST /api/predict/benchmark/batch`.
