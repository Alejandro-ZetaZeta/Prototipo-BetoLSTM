# Elbeto Deployment

## 1. Prepare Ubuntu

Install Docker Engine and the Compose plugin, then create the application directories:

```bash
sudo mkdir -p /srv/elbeto/models
sudo chown -R "$USER":"$USER" /srv/elbeto
cd /srv/elbeto
```

Clone the repository into `/srv/elbeto/source/Prototipo-BetoLSTM`:

```bash
git clone https://github.com/YOUR_USER/YOUR_REPOSITORY.git /srv/elbeto/source/Prototipo-BetoLSTM
cd /srv/elbeto/source/Prototipo-BetoLSTM
cp .env.example .env
```

Edit `.env` and set `ALLOWED_ORIGINS` to the deployed Astro URL. Add your ngrok token and assigned ngrok domain. Do not commit `.env`.

## 2. Transfer model artifacts

From the development computer, copy the model directory contents to Ubuntu:

```powershell
scp -r .\backend\models\* anthony@SERVER_IP:/srv/elbeto/models/
```

The server must contain this layout:

```text
/srv/elbeto/models/
├── beto_rf_rnf/
├── beto_lstm_rf_rnf/
├── random_forest_pipeline.joblib
└── naive_bayes_pipeline.joblib
```

Do not copy the LSTM ZIP if its directory is already extracted.

## 3. Configure ngrok

Create or sign in to an ngrok account and copy the authtoken from the ngrok dashboard.
The free plan provides one assigned development domain. Copy that complete HTTPS URL.

On Ubuntu, edit the local environment file:

```bash
cd /srv/elbeto/source/Prototipo-BetoLSTM
nano .env
```

Set these values:

```env
NGROK_AUTHTOKEN=your_real_ngrok_token
NGROK_DOMAIN=https://your-assigned-domain.ngrok-free.app
```

Keep the token only in `.env`. Compose passes it to the ngrok container without storing it in the repository.
The service target is `api:8000`; `api` is the Compose service name.

## 4. Start services

From the repository directory:

```bash
cd /srv/elbeto/source/Prototipo-BetoLSTM
docker compose up -d --build
docker compose logs -f api
```

The API is not published directly to the host. ngrok is the only public route.

Check the public API:

```bash
curl https://your-assigned-domain.ngrok-free.app/health
```

Expected response includes:

```json
{"status":"ok","model_loaded":true,"beto_lstm_loaded":true}
```

Useful maintenance commands:

```bash
docker compose ps
```

## 5. Deploy frontend

Build the Astro site with this environment variable in Vercel:

```text
PUBLIC_API_URL=https://your-assigned-domain.ngrok-free.app
```

Add the resulting Vercel URL to `ALLOWED_ORIGINS` on Ubuntu, then restart the API:

```bash
nano /srv/elbeto/source/Prototipo-BetoLSTM/.env
docker compose up -d --build --force-recreate api
```

## Notes

- Run one Uvicorn worker; multiple workers duplicate model memory.
- Keep model files outside Git and outside the Docker image.
- The ngrok token and `.env` are secrets.
- The ngrok development domain stays stable for your account; the Ubuntu ngrok container restarts automatically with Compose.
- The ngrok free plan has request and bandwidth limits and may show a browser interstitial; the frontend sends `ngrok-skip-browser-warning`.
- The Ubuntu host must remain powered on and connected for the API to be available.
