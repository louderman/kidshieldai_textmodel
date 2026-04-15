# KidShield AI

A content classification system consisting of a FastAPI backend and a Chrome extension. It detects and blurs unsafe text, images, and flags suspicious links on any webpage.

## Project Structure

```
kidshieldai_textmodel/
├── kidshield_backend/    # fastAPI  server
│   ├── models/
│   │   ├── mdeberta-kidshield/   # text classification model
│   │   └── vit/                  # image classification model
│   ├── app/
│   └── run.py
├── kidshield_chrome/     # frontend chrome extension 
└── testpage/             # test page
```

---

## 1. Running the Backend

### First-time setup

Install Python dependencies globally:

For powershell use:
py -m pip install -r kidshield_backend/requirements.txt


### Start the server

For powershell use:
cd kidshield_backend
py run.py


The API will be available at `http://127.0.0.1:8000`.

Download model files first
`kidshield_backend/models/mdeberta-kidshield/` — text model
`kidshield_backend/models/vit/kidshield_vit_scripted.pt` — image model

---

## 2. Running the Chrome Extension

### First-time setup

For powershell use:
cd kidshield_chrome
npm install
npm run build

### Load in Chrome

1. Go to `chrome://extensions/`
2. Enable Developer mode (top right toggle)
3. Click Load unpacked
4. Select the `kidshield_chrome/dist/` folder
5. Click Details on KidShield AI and enable Allow access to file URLs

### Rebuild after changes

For powershell use:
cd kidshield_chrome
npm run build

Then click the refresh icon on the extension in `chrome://extensions/`
if refresh doesnt work, just remove and add it again.

---

## What it does

| Content type | Detection | Action |
|---|---|---|
| Unsafe text | DeBERTa ML model + keyword blocklist | Blurs text — click to reveal |
| Leet-speak (`sh1t`) | Decoded before classification | Same as above |
| Unsafe images | ViT ML model | Blurs image — click to reveal |
| Suspicious links | ML model + domain heuristics | Red strikethrough + warning badge |

---

## Notes

- The backend must be running before the extension can classify content.
- Do not commit `node_modules/` to git.
