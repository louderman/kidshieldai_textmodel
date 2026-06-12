# KidShield AI

A child safety content filtering system consisting of a FastAPI backend and a Chrome extension. It scans webpages in real time and detects unsafe text, images, and suspicious links using two custom-trained ML models.

| Content type | Detection method | Browser action |
|---|---|---|
| Unsafe text | DeBERTa ML model + keyword blocklist | Blurs text — click to reveal |
| Leet-speak (`sh1t`, `@ss`) | Decoded before classification | Same as above |
| Unsafe images | Vision Transformer (ViT) ML model | Blurs image — click to reveal |
| Suspicious links | ML heuristics + domain pattern matching | Red strikethrough + warning badge |

---

## Project Structure

```
kidshieldai_textmodel/
├── kidshield_backend/          # FastAPI server
│   ├── app/
│   │   ├── main.py             # App entrypoint + lifespan
│   │   ├── core/
│   │   │   ├── config.py       # Settings (host, port, model paths)
│   │   │   ├── model.py        # Text classification (DeBERTa)
│   │   │   └── image_model.py  # Image classification (ViT)
│   │   └── api/routes/         # /health, /classify, /classify-image
│   ├── models/                 # Model weights — see below
│   ├── requirements.txt
│   └── run.py                  # Server entry point
├── kidshield_chrome/           # Chrome extension (React + Vite)
│   ├── src/
│   │   ├── background/         # Message relay to backend
│   │   └── content/            # DOM scanner + blur logic
│   ├── public/manifest.json
│   └── package.json
└── testpage/                   # Static HTML page for manual testing
```

---

## Getting the Code

Clone from the **`all`** branch:

```powershell
git clone --branch all <repo-url>
```

The `all` branch contains the latest stable, production-ready version of the project.

---

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| Google Chrome | Any recent version |
| ~2 GB free disk space | For model weights |

---

## Step 1 — Place the Model Files

The backend requires two model files that are **not included in the repository** due to their size. You must obtain them separately (from the training repo, shared drive, or the handing-off team) and place them at the exact paths below.

### Text model (DeBERTa)

The text model is stored in the **`main`** branch of this repository. If you cloned the `all` branch, switch to `main` and copy the model files out, or download the `main` branch separately:

```powershell
git clone --branch main <this-repo-url>
```

Place the full model directory at:

```
kidshield_backend/models/mdeberta-kidshield/
```

The directory must contain at minimum:

```
mdeberta-kidshield/
├── config.json
├── tokenizer_config.json
├── spm.model
├── special_tokens_map.json
├── added_tokens.json
└── model.safetensors        ← ~1.1 GB
```

### Image model (Vision Transformer)

The image model lives in a separate repository. Clone its **`main`** branch:

```powershell
git clone --branch main https://github.com/Meherbob2285-dot/Kidshield-Visual-Model
```

Then place the scripted model file at:

```
kidshield_backend/models/vit/kidshield_vit_scripted.pt    ← ~344 MB
```

### Verify placement

After placing both files, your `models/` directory should look like this:

```
kidshield_backend/models/
├── mdeberta-kidshield/
│   ├── config.json
│   ├── model.safetensors
│   └── ... (other tokenizer files)
└── vit/
    └── kidshield_vit_scripted.pt
```

> **Note:** The model paths are configurable via environment variables or a `.env` file — see the [Configuration](#configuration) section if you need to store them elsewhere.

---

## Step 2 — Backend Setup

### Install Python dependencies

From the repository root:

```powershell
py -m pip install -r kidshield_backend/requirements.txt
```

### Start the server

```powershell
cd kidshield_backend
py run.py
```

The API will be available at `http://127.0.0.1:8000`.

You should see output confirming both models loaded successfully. If either model is missing, the server will print an error indicating which path it expected.

---

## Step 3 — Chrome Extension Setup

### Install dependencies and build

```powershell
cd kidshield_chrome
npm install
npm run build
```

This produces the bundled extension in `kidshield_chrome/dist/`.

### Load the extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `kidshield_chrome/dist/` folder
5. Click **Details** on the KidShield AI card and enable **Allow access to file URLs**

### Rebuild after making changes

```powershell
cd kidshield_chrome
npm run build
```

Then click the refresh icon on the extension card in `chrome://extensions/`. If the refresh does not take effect, remove the extension and load it again.

---

## Running Order

The backend **must be running before** the Chrome extension can function. Always start the backend first.

1. Start the backend (`py run.py` inside `kidshield_backend/`)
2. Load the Chrome extension
3. Open any webpage — the extension will automatically scan and blur content

---

## Configuration

The backend reads settings from environment variables or an optional `.env` file placed in `kidshield_backend/`.

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Server bind address |
| `PORT` | `8000` | Server port |
| `MODEL_PATH` | `<backend_dir>/models/mdeberta-kidshield` | Path to the DeBERTa model directory |
| `VIT_MODEL_PATH` | `<backend_dir>/models/vit/kidshield_vit_scripted.pt` | Path to the ViT `.pt` file |
| `DEBUG` | `False` | Enable debug mode |

Example `.env` file (optional):

```
MODEL_PATH=/data/models/mdeberta-kidshield
VIT_MODEL_PATH=/data/models/vit/kidshield_vit_scripted.pt
PORT=8080
```

> **Chrome extension note:** The backend URL is hardcoded to `http://127.0.0.1:8000` in `src/background/background.js`. If you change the backend host or port, update that file and rebuild the extension.

---

## API Reference

| Method | Endpoint | Request body | Response |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/classify` | `{"text": "..."}` | `{"label": "safe"\|"unsafe", "score": 0.0–1.0}` |
| `POST` | `/classify-image` | `{"image": "<base64>"}` | `{"label": "benign"\|"unsafe", "score": 0.0–1.0}` |

---

## Manual Testing

A static test page is included at `testpage/`. Open it directly in Chrome (with the extension loaded and backend running) to verify the extension is blurring content correctly.

---

## Troubleshooting

**Backend fails to start with a model-not-found error**
The model files are missing or placed at the wrong path. Re-check Step 1 and confirm the exact directory names match.

**Extension is not blurring any content**
Confirm the backend is running (`http://127.0.0.1:8000/health` should return `{"status":"ok"}`). If you changed the port, rebuild the extension after updating `src/background/background.js`.

**`torch` install is very slow or fails**
PyTorch is a large package. If `pip install` is slow, try installing torch separately first from [pytorch.org](https://pytorch.org/get-started/locally/) using the command appropriate for your OS and CUDA version, then run `pip install -r requirements.txt` again.
