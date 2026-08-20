# Elbeto Deployment

## 1. Prepare Ubuntu

Install Docker Engine and the Compose plugin, then create the application directories:

```bash
sudo mkdir -p /srv/elbeto/models
sudo chown -R "$USER":"$USER" /srv/elbeto
cd /srv/elbeto
```

Clone the repository into `/srv/elbeto/source`:

```bash
git clone https://github.com/YOUR_USER/YOUR_REPOSITORY.git source
cd source
cp .env.example .env
```

Edit `.env` and set `ALLOWED_ORIGINS` to the deployed Astro URL. Do not commit `.env`.

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

## 3. Configure Cloudflare Tunnel

Install `cloudflared` on a workstation with access to the Cloudflare account:

```bash
cloudflared tunnel login
cloudflared tunnel create elbeto-api
cloudflared tunnel route dns elbeto-api api.example.com
```

Copy the generated credentials file to the Ubuntu server:

```bash
scp ~/.cloudflared/YOUR_TUNNEL_UUID.json anthony@SERVER_IP:/srv/elbeto/source/cloudflared/
```

On Ubuntu, copy the example configuration and replace the UUID and hostname:

```bash
cd /srv/elbeto/source
cp cloudflared/config.example.yml cloudflared/config.yml
nano cloudflared/config.yml
```

The service target must remain `http://api:8000`; `api` is the Compose service name.

## 4. Start services

From the repository directory:

```bash
cd /srv/elbeto/source
docker compose up -d --build
docker compose logs -f api
```

The API is not published directly to the host. Cloudflare Tunnel is the only public route.

Check the public API:

```bash
curl https://api.example.com/health
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
PUBLIC_API_URL=https://api.example.com
```

Add the resulting Vercel URL to `ALLOWED_ORIGINS` on Ubuntu, then restart the API:

```bash
nano .env
docker compose up -d --build --force-recreate api
```

## Notes

- Run one Uvicorn worker; multiple workers duplicate model memory.
- Keep model files outside Git and outside the Docker image.
- Cloudflare Tunnel credentials and `.env` are secrets.
- The Ubuntu host must remain powered on and connected for the API to be available.
