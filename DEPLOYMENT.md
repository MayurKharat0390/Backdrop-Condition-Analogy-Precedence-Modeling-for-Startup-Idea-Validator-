# BCAPM Cloud Deployment Guide

This repository contains all production configurations needed to deploy the **BCAPM Venture Underwriting Terminal** to free cloud hosting platforms.

---

## Option 1: 1-Click Deployment on Render (100% Free)

**Render** provides free web service hosting directly connected to your GitHub repository.

### Steps:
1. Push your latest code to your GitHub repository:
   ```bash
   git add .
   git commit -m "Deploy BCAPM Institutional Venture Terminal"
   git push origin main
   ```
2. Go to [dashboard.render.com](https://dashboard.render.com/) and sign in with GitHub.
3. Click **New +** → **Web Service**.
4. Connect your repository: `Backdrop-Condition-Analogy-Precedence-Modeling-for-Startup-Idea-Validator-`.
5. Render will automatically detect `render.yaml` or you can configure:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app/web_app.py`
   - **Plan**: `Free`
6. Click **Create Web Service**.
7. In ~2 minutes, your live public URL will be ready:
   `https://bcapm-terminal.onrender.com`

---

## Option 2: 1-Click Deployment on Railway

**Railway** provides instant deployment with zero configuration:

### Steps:
1. Go to [railway.app](https://railway.app/) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your repository.
4. Railway will automatically detect the `Procfile` / `Dockerfile` and deploy the service.
5. In **Settings** → **Networking**, click **Generate Domain**.
6. Your live public URL will be generated instantly (e.g., `https://bcapm-production.up.railway.app`).

---

## Option 3: Deployment on Hugging Face Spaces (Free Cloud Hosting)

Hugging Face Spaces provides permanent free hosting with Docker or Python apps:

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space Name: `bcapm-terminal`.
3. Select Space SDK: **Docker** (Blank).
4. Connect or push this repository using the provided Hugging Face git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/bcapm-terminal
   git push hf main
   ```
5. Hugging Face will automatically build using the included [`Dockerfile`](Dockerfile) and launch your public terminal!

---

## Option 4: Local Production Run

Run the terminal locally on any custom port:
```bash
python app/web_app.py --port 8080
```
Open `http://localhost:8080` in your web browser.
