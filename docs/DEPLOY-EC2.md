# Deploying on AWS EC2 (g6, NVIDIA L4 24 GB, Ubuntu 22.04)

A test deployment of the whole service on one GPU instance: the API, PostgreSQL, Ollama (`qwen3:8b`,
`qwen3-vl:8b-instruct`) and the experimental handwriting model. It uses `docker-compose.yml` +
`docker-compose.gpu.yml`.

What runs where:

| Part | Runs on |
|---|---|
| PaddleOCR (printed text) | **CPU** (the image has the CPU build of PaddlePaddle) |
| Ollama: text LLM + vision fallback | GPU |
| Handwriting model (`/v1/handwriting`) | GPU (bfloat16) |
| PostgreSQL | CPU, `postgres-data` volume |

Because OCR runs on the CPU, the instance size matters as much as the GPU. g6.xlarge, 2xlarge, 4xlarge and 8xlarge
all have the same single L4; they differ in CPU and RAM.

| Instance | vCPU / RAM | |
|---|---|---|
| g6.xlarge | 4 / 16 GB | too small: OCR is slow and RAM is tight |
| **g6.2xlarge** | 8 / 32 GB | good for testing |
| g6.4xlarge | 16 / 64 GB | more OCR throughput (raise `OCR_ENGINES` in `.env`) |

These files were written without access to a GPU machine. **Nothing here has been run on a g6 yet.** Check each step's
output as you go.

## 1. Launch the instance (AWS console)

1. **GPU quota.** New accounts often have a quota of 0 for GPU instances. Open *Service Quotas → Amazon EC2 →
   "Running On-Demand G and VT instances"* and make sure it is at least the vCPU count of your size (8 for g6.2xlarge).
   Request an increase if not. It can take a while to be approved.
2. **AMI.** Either one works:
   - **Recommended:** *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*. Find it in the AMI catalog by
     searching for "Deep Learning Base OSS Nvidia Driver GPU AMI Ubuntu 22.04". It comes with the NVIDIA driver and
     usually Docker and the container toolkit too.
   - plain *Ubuntu Server 22.04 LTS*. Step 3 installs the driver (one extra reboot).
3. **Instance type:** g6.2xlarge (see above).
4. **Key pair:** create one or pick one, and keep the `.pem` file.
5. **Storage:** **150 GiB gp3** for the root volume. The images (CUDA torch, Paddle), the Ollama models (~12 GB), the
   handwriting model (~7.5 GB) and PostgreSQL don't fit in the default size.
6. **Security group:** inbound **SSH (22) from "My IP" only**. Do **not** open port 8000: the API is published on
   the server's `127.0.0.1` only, and you reach it through an SSH tunnel (step 7).

## 2. Connect and get the code

From your laptop (PowerShell). If ssh complains that the key's permissions are too open, restrict the file first:
```powershell
icacls C:\path\to\key.pem /inheritance:r /grant:r "$($env:USERNAME):(R)"
ssh -i C:\path\to\key.pem ubuntu@<public-ip>
```
On the server:
```bash
git clone -b <your-branch> <your-repo-url> ocr     # private repo: GitHub user name + a personal access token
cd ocr
```

## 3. Prepare the host (driver, Docker, NVIDIA Container Toolkit)

```bash
bash scripts/ec2_setup.sh
```
- If it installs the driver, it says so and stops. Run `sudo reboot`, reconnect, `cd ocr`, and run it again.
- It ends by running `nvidia-smi` **inside a container**. You should see the "NVIDIA L4" with about 23 GB of memory.
  The "CUDA Version" at the top right must be **12.6 or higher** (the handwriting model's torch is built for CUDA
  12.6).
- If it added you to the `docker` group, run `newgrp docker` (or log out and back in) so `docker` works without sudo.

## 4. Create `.env`

```bash
python3 scripts/make_env.py
```
This writes `.env` (mode 600) from `.env.example` with new secrets: `API_KEYS`, `POSTGRES_PASSWORD`,
`RESULT_ENCRYPTION_KEY`. It also sets `COMPOSE_FILE` so every `docker compose` command uses the GPU file, and turns
the handwriting endpoint on with `HANDWRITING_DEVICE=cuda`. **Copy the API key it prints.** You need it for every
request. It is also in `.env` (`grep ^API_KEYS .env`).

Keep `.env` on the server and back it up somewhere safe. If you lose `RESULT_ENCRYPTION_KEY`, the stored results can't
be read. Running `make_env.py --force` later replaces every secret. Only do that before the database exists, or after
`docker compose down -v`, which deletes everything.

## 5. Download the models, build, start

```bash
docker compose --profile setup run --rm ollama-pull          # qwen3:8b + qwen3-vl:8b-instruct (one time)
docker compose build                                         # first build is slow: Paddle, CUDA torch, OCR models
docker compose --profile setup run --rm handwriting-pull     # handwriting model, ~7.5 GB (one time, after build)
docker compose up -d
```
Only `ollama-pull`, `handwriting-pull` and the build need the internet. At runtime, Ollama and PostgreSQL have no
internet route, and the app runs with `HF_HUB_OFFLINE=1`.

## 6. Check it

```bash
docker compose ps                                  # app "healthy" after ~1-2 min; postgres healthy; ollama up
docker compose logs --tail 50 app                  # a configuration problem shows up here as "refusing to start: ..."
curl -s http://127.0.0.1:8000/health; echo         # "status":"ok"; llm.textModel, llm.visionModel, ocr.modelsPresent true
curl -s http://127.0.0.1:8000/health/handwriting; echo   # enabled, depsInstalled, modelDownloaded: true
```
Send one document (step 7). Then check that the models really are on the GPU:
```bash
docker compose exec ollama ollama ps               # PROCESSOR column: "100% GPU" (not "CPU" or a split)
nvidia-smi                                         # memory in use; the handwriting model adds ~8 GB once loaded
```

## 7. Test from your laptop

Open a tunnel and leave that window open:
```powershell
ssh -i C:\path\to\key.pem -N -L 8000:127.0.0.1:8000 ubuntu@<public-ip>
```
In another PowerShell window:
```powershell
$KEY = "<the API key from step 4>"
curl.exe -H "X-Api-Key: $KEY" -F "file=@D:\private\scan.pdf" http://127.0.0.1:8000/v1/extract        # text only
curl.exe -H "X-Api-Key: $KEY" -F "file=@D:\private\id.jpg"  http://127.0.0.1:8000/v1/documents      # -> jobId
curl.exe -H "X-Api-Key: $KEY" "http://127.0.0.1:8000/v1/jobs/<jobId>?wait=30"
curl.exe -H "X-Api-Key: $KEY" -F "file=@D:\private\note.jpg" http://127.0.0.1:8000/v1/handwriting
```
The handwriting model loads on the first `/v1/handwriting` request, so that request is slower than the ones after it.

## 8. Admin dashboard (`/admin`)

`/admin` answers 404 until `ADMIN_USERS_FILE` is set. The users file holds only scrypt password hashes. Put it in
`config/local/` on the server: that folder is git-ignored and mounted into the container (`docker-compose.yml`).

```bash
cd ~/ocr
# 1. a password hash (asks for the password, at least 12 characters; nothing is stored yet)
docker compose exec app uv run --no-sync python -c   "import getpass; from app.admin.auth import hash_password; print(hash_password(getpass.getpass('Password: ')))"

# 2. the users file: paste the hash printed above
nano config/local/admins.yaml
```
```yaml
admins:
  admin:
    name: "Admin"
    role: admin
    password_scrypt: "scrypt$16384$8$1$...the hash..."
```
```bash
# 3. turn it on (.env from make_env.py already has ADMIN_SESSION_SECRET; older ones need it added)
grep -q '^ADMIN_SESSION_SECRET=' .env || echo "ADMIN_SESSION_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
echo 'ADMIN_USERS_FILE=/app/config/local/admins.yaml' >> .env
docker compose up -d
```
Open **http://localhost:8000/admin** through the SSH tunnel (step 7). Use `localhost`: in production mode the
session cookie is `Secure`, and browsers accept that over plain HTTP only for `localhost`. Adding or changing users
in `admins.yaml` later needs no restart, because the file is re-read when it changes.

## 9. Day to day

| Task | Command |
|---|---|
| Deploy new code | `git pull && docker compose up -d --build` |
| Logs | `docker compose logs -f app` |
| Stop (keeps data and models) | `docker compose down` |
| Start again | `docker compose up -d` |
| **Delete everything** (database, models) | `docker compose down -v`. Only if you mean it |

**Stop the EC2 instance** when you are not testing: a running g6 costs money every hour. A stopped instance only costs
its disk. The data, models and `.env` survive a stop and start. The public IP changes unless you attach an Elastic IP.

## 10. Testing the Fanar handwriting model

Next to the current handwriting model (`sherif-v3`), the server can offer `fanar`: a LoRA adapter on
`QCRI/Fanar-2-Oryx-IVU`, loaded in 4-bit, reading one line at a time (README, "Several models side by side"). The
current model stays the default; a request uses Fanar only with `-F model=fanar`. Switching back needs no code
change: remove the `HANDWRITING_MODELS_FILE` line from `.env` and run `docker compose up -d`.

```bash
cd ~/ocr && git pull                     # the branch with the Fanar support
echo 'HANDWRITING_MODELS_FILE=config/handwriting_models.yaml' >> .env
grep -q '^OLLAMA_MAX_LOADED_MODELS=' .env \
  && sed -i 's|^OLLAMA_MAX_LOADED_MODELS=.*|OLLAMA_MAX_LOADED_MODELS=1|' .env \
  || echo 'OLLAMA_MAX_LOADED_MODELS=1' >> .env   # leave GPU memory for both handwriting models while comparing
docker compose build                      # new packages: peft, bitsandbytes
docker compose --profile setup run --rm handwriting-pull   # downloads Fanar: ~16.6 GB base + 162 MB adapter
docker compose up -d
curl -s http://127.0.0.1:8000/health/handwriting; echo    # models.fanar: depsInstalled, modelDownloaded, adapterDownloaded true
```
GPU memory, rough estimates to check with `nvidia-smi`: the current model ~8 GB, Fanar in 4-bit ~6 GB, one Ollama
model ~6–7 GB. Each handwriting model loads on its first request and then stays loaded.

Try it (through the SSH tunnel, from your laptop):
```powershell
curl.exe -H "X-Api-Key: $KEY" -F "file=@D:\private\page.jpg" -F model=fanar http://127.0.0.1:8000/v1/handwriting
curl.exe -H "X-Api-Key: $KEY" -F "file=@D:\private\page.jpg" http://127.0.0.1:8000/v1/handwriting   # current model
# an image that is already ONE line: send it whole instead of splitting it
curl.exe -H "X-Api-Key: $KEY" -F "file=@D:\private\line.png" -F model=fanar -F segmenter=whole_page http://127.0.0.1:8000/v1/handwriting
```
The first Fanar request loads the model (slow); then check `docker compose logs app | grep "handwriting model loaded"`
says `quantization=4bit adapter=True`. Look at the page `warnings`: `no_lines` (the detector found no line),
`too_many_lines`, `truncated`, `repetition_loop`.

**Compare both on the same hand-checked samples** (README, "Evaluate it on your documents": a folder with
`manifest.csv` and typed transcripts; copy it to the server, e.g. `~/hw-samples`). The evaluation runs inside the app
image, in its own process, so stop the app first (both would load models on the GPU):
```bash
cd ~/ocr
mkdir -p eval/out && chmod 777 eval/out   # the container's user writes the reports here
docker compose stop app
for m in sherif-v3 fanar; do
  docker compose run --rm --no-deps -v "$PWD/eval:/app/eval" -v "$HOME/hw-samples:/samples:ro" app \
    uv run --no-sync python -m eval.run_handwriting_eval /samples --model "$m" --save-text
done
docker compose start app
ls eval/out/                              # handwriting-<time>-sherif-v3/ and handwriting-<time>-fanar/: report.md each
```
Compare raw and normalized CER/WER per category, the invented text on blank / signature / stamp pages, and the time
per page. The reports hold your file names and (with `--save-text`) the read text: keep `eval/out` private.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not select device driver "nvidia"` | The container toolkit is missing or not configured: run `bash scripts/ec2_setup.sh` again |
| app keeps restarting | `docker compose logs app`: the "refusing to start" line lists every unsafe setting |
| `/health` 503, `textModel: false` | Ollama models missing (step 5 `ollama-pull`), or Ollama still starting |
| `/v1/handwriting` 503 `handwriting_unavailable` | The message says why. `modelDownloaded: false` means run `handwriting-pull`. "no CUDA device" means the app container has no GPU: check `COMPOSE_FILE` in `.env` and `docker compose config` |
| `ollama ps` shows CPU or a split | Not enough free VRAM: check `nvidia-smi`; keep `OLLAMA_MAX_LOADED_MODELS=1` |
| Build fails | Paste the error. The Docker files haven't been built on a GPU host before this deployment |
| Fanar: "adapter's weights were not applied (0 of N ...)" | The adapter's layer names don't match the model as this transformers/peft version builds it (it refuses rather than run the base model silently). The old Qwen2.5-VL layout (`model.layers.N`, which the Fanar adapter uses) is mapped on load since this was fixed: `git pull`, `docker compose build app`, `docker compose up -d`. Still there: send the log line |
| Fanar: `4bit needs a CUDA GPU`, or a bitsandbytes error at load | The app container has no GPU (see above), or bitsandbytes doesn't fit this CUDA/driver: send the log (`docker compose logs app`) |
| `400 unknown_model` | `HANDWRITING_MODELS_FILE` isn't set in `.env` (only one model then), or a typo in `model` |
