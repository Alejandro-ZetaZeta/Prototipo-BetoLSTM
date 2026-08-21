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

The page uses `http://127.0.0.1:8000` by default. Set `PUBLIC_API_URL` when using an API hosted elsewhere. This value is not displayed in the interface.

For a local API:

```powershell
$env:PUBLIC_API_URL = "http://127.0.0.1:8000"
npm run dev
```

For the Ubuntu API exposed through ngrok:

```powershell
$env:PUBLIC_API_URL = "https://your-assigned-domain.ngrok-free.app"
npm run dev
```

Frontend does not persist submitted text.

Train baseline artifacts from project root:

```powershell
python training/train_baselines.py
```
