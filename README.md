# 🎯 Clickbait Detector — MLOps-Ready NLP Project

A production-grade, lightweight clickbait detection system built for **AWS free tier** deployment.
No PyTorch, no Transformers — just scikit-learn + FastAPI + MLflow + DVC.

```
clickbait_detector/
├── data/
│   ├── clickbait_data.csv        ← 100-row labelled dataset
│   └── clickbait_data.csv.dvc   ← DVC tracking file
├── src/
│   ├── preprocess.py             ← Text cleaning
│   ├── features.py               ← TF-IDF + handcrafted features
│   ├── train.py                  ← Training + MLflow logging
│   ├── predict.py                ← Inference module
│   └── monitor.py                ← Logging + drift detection
├── tests/
│   └── test_pipeline.py          ← Unit + integration tests
├── models/                       ← Saved pkl artefacts (git-ignored, DVC-tracked)
├── logs/                         ← Runtime logs
├── .github/workflows/ci.yml      ← GitHub Actions CI/CD
├── dvc.yaml                      ← DVC pipeline
├── params.yaml                   ← Hyperparameters
├── app.py                        ← FastAPI server
├── Dockerfile                    ← Container for AWS EC2
└── requirements.txt
```

---

## ✅ PART 1 — Local Setup (15 min)

### Prerequisites
- Python 3.9 or 3.10
- Git
- pip

### Step 1 — Clone / Unzip and Enter Project

```bash
# If using the zip file:
unzip clickbait_detector.zip
cd clickbait_detector

# Or if using git:
git clone <your-repo-url>
cd clickbait_detector
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ Takes ~2 minutes. Total install size: ~300 MB.

### Step 4 — Train the Model

```bash
python src/train.py
```

Expected output:
```
INFO  Loading data from data/clickbait_data.csv
INFO  Loaded 100 rows | Clickbait: 50 | Not: 50
INFO  Model saved to models/classifier.pkl
=== Training Complete ===
  accuracy: 0.95
  f1: 0.95
  cv_mean_accuracy: 0.91
```

### Step 5 — Run Tests

```bash
pytest tests/ -v
```

All tests should pass. End-to-end tests require the trained model (auto-detected).

### Step 6 — Start the API Server

```bash
python app.py
# OR:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 7 — Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"headline": "You Won'\''t Believe What Happened Next!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"headlines": ["Shocking secret revealed!", "Parliament passes new bill"]}'

# Monitoring stats
curl http://localhost:8000/stats

# Interactive Swagger docs
open http://localhost:8000/docs
```

---

## 📊 PART 2 — MLflow Experiment Tracking

### View the MLflow Dashboard

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` in your browser.

You'll see:
- Experiment: `clickbait-detection`
- Logged parameters: C, test_size, max_features, ngram_range, etc.
- Logged metrics: accuracy, precision, recall, f1, cv scores

### Run Multiple Experiments

```bash
# Edit params.yaml, then retrain:
# Change C: 1.0 → C: 0.1, then:
python src/train.py

# Or change ngram_range and retrain:
# ngram_range: [1, 3]
python src/train.py
```

Each run appears as a separate row in MLflow UI.

---

## 📦 PART 3 — DVC Data Versioning

### Initialize DVC (first time only)

```bash
git init   # if not already a git repo
dvc init
git add .dvc/
git commit -m "Initialize DVC"
```

### Track the Dataset

```bash
dvc add data/clickbait_data.csv
git add data/clickbait_data.csv.dvc data/.gitignore
git commit -m "Track dataset with DVC"
```

### Configure DVC Remote (S3 — free tier eligible)

```bash
# Create an S3 bucket in AWS Console first, then:
dvc remote add -d myremote s3://your-bucket-name/dvc-store
dvc remote modify myremote region ap-south-1   # or your region

# Push data to S3
dvc push

# On a new machine, pull data:
dvc pull
```

### Run the Full DVC Pipeline

```bash
dvc repro
# This runs: preprocess → train, in order, only re-running changed stages
```

---

## 🐙 PART 4 — GitHub Setup (10 min)

### Step 1 — Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name it `clickbait-detector`
3. Choose **Private** (recommended for ML projects)
4. **Do NOT** initialize with README (you already have one)
5. Click **Create repository**

### Step 2 — Push Your Code

```bash
cd clickbait_detector

git init
git add .
git commit -m "Initial MLOps project commit"

git remote add origin https://github.com/YOUR_USERNAME/clickbait-detector.git
git branch -M main
git push -u origin main
```

### Step 3 — Set Up GitHub Secrets (for AWS deployment)

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:
| Secret Name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret key |
| `AWS_DEFAULT_REGION` | e.g. `ap-south-1` |

### Step 4 — Verify CI/CD Pipeline

Push any change to trigger the pipeline:
```bash
echo "# trigger CI" >> README.md
git add README.md
git commit -m "Trigger CI pipeline"
git push
```

Go to **Actions** tab on GitHub to watch the pipeline:
- ✅ Install dependencies
- ✅ Run unit tests
- ✅ Train model
- ✅ Run full tests + metric validation
- ✅ Upload model artifacts
- ✅ Docker build (main branch only)

---

## ☁️ PART 5 — AWS EC2 Deployment (20 min)

### Step 1 — Launch EC2 Instance

1. Go to [AWS Console → EC2](https://console.aws.amazon.com/ec2)
2. Click **Launch Instance**
3. **Name:** `clickbait-detector`
4. **AMI:** Ubuntu Server 22.04 LTS (Free tier eligible ✅)
5. **Instance type:** `t2.micro` (Free tier — 1 vCPU, 1 GB RAM)
6. **Key pair:** Create new → name it `clickbait-key` → Download `.pem` file
7. **Security group:** Allow inbound:
   - SSH (port 22) from your IP
   - Custom TCP (port 8000) from Anywhere (0.0.0.0/0)
8. **Storage:** 8 GB gp2 (default, free tier)
9. Click **Launch Instance**

### Step 2 — Connect to EC2

```bash
# Make key read-only (Linux/macOS)
chmod 400 ~/Downloads/clickbait-key.pem

# SSH into instance
ssh -i ~/Downloads/clickbait-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

Find your Public IP in EC2 Console → Instances → your instance.

### Step 3 — Set Up EC2 Environment

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.10 and tools
sudo apt-get install -y python3.10 python3.10-venv python3-pip git unzip

# Verify Python
python3 --version   # Should show 3.10.x
```

### Step 4 — Deploy the Project

**Option A — Direct file copy (simplest)**
```bash
# On your LOCAL machine:
zip -r clickbait_detector.zip clickbait_detector/ \
  --exclude "*.pyc" "*/__pycache__/*" "*/.venv/*" "*/mlruns/*"

scp -i ~/Downloads/clickbait-key.pem \
  clickbait_detector.zip \
  ubuntu@YOUR_EC2_PUBLIC_IP:~/

# Back on EC2:
unzip clickbait_detector.zip
cd clickbait_detector
```

**Option B — Git clone (if you pushed to GitHub)**
```bash
# On EC2:
git clone https://github.com/YOUR_USERNAME/clickbait-detector.git
cd clickbait-detector
```

### Step 5 — Install and Train on EC2

```bash
cd clickbait_detector

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (~3 min on t2.micro)
pip install --upgrade pip
pip install -r requirements.txt

# Train model
python src/train.py
```

### Step 6 — Run the API Server

```bash
# Run in foreground (for testing):
python app.py

# Run in background with nohup (keeps running after SSH disconnect):
nohup uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 > logs/server.log 2>&1 &

# Check it's running:
curl http://localhost:8000/health
```

### Step 7 — Test from Your Local Machine

```bash
# Replace with your EC2 Public IP:
curl http://YOUR_EC2_PUBLIC_IP:8000/health

curl -X POST http://YOUR_EC2_PUBLIC_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"headline": "You Won'\''t Believe What Happened"}'
```

### Step 8 — Keep Server Running (systemd service)

```bash
# Create service file on EC2:
sudo tee /etc/systemd/system/clickbait.service << 'EOF'
[Unit]
Description=Clickbait Detector API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/clickbait_detector
Environment=PATH=/home/ubuntu/clickbait_detector/.venv/bin
ExecStart=/home/ubuntu/clickbait_detector/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start:
sudo systemctl daemon-reload
sudo systemctl enable clickbait
sudo systemctl start clickbait
sudo systemctl status clickbait
```

---

## 🐳 PART 6 — Docker Deployment (Optional)

```bash
# Build image
docker build -t clickbait-detector:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  --name clickbait \
  --restart unless-stopped \
  clickbait-detector:latest

# Check logs
docker logs clickbait

# Test
curl http://localhost:8000/health
```

---

## 🔧 Hyperparameter Tuning

Edit `params.yaml` and run `python src/train.py` or `dvc repro`:

```yaml
train:
  C: 0.5          # Lower = more regularisation
  max_features: 3000   # Fewer TF-IDF features = faster
  ngram_range: [1, 3]  # Add trigrams
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + model readiness |
| `/predict` | POST | Single headline prediction |
| `/predict/batch` | POST | Up to 100 headlines |
| `/stats` | GET | Rolling prediction stats |
| `/drift` | GET | Drift detection check |
| `/docs` | GET | Swagger UI |

### Example Response

```json
{
  "headline": "You Won't Believe What Happened Next!",
  "is_clickbait": true,
  "label": 1,
  "label_text": "Clickbait",
  "confidence": 0.9731
}
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Only unit tests (no model required)
pytest tests/ -v -k "not EndToEnd"

# With coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: preprocess` | Run from project root: `python src/train.py` |
| `Model not found` error on API start | Run `python src/train.py` first |
| Port 8000 blocked on EC2 | Check Security Group inbound rules allow port 8000 |
| `pip install` slow on t2.micro | Normal — takes ~3 min. Use `--no-cache-dir` to save disk |
| MLflow UI shows no runs | Run `mlflow ui` from project root where `mlruns/` exists |
| DVC push fails | Configure AWS credentials: `aws configure` |

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~0.95 |
| F1 Score | ~0.95 |
| CV Mean Accuracy | ~0.91 |
| Model Size | < 2 MB |
| Inference Time | < 5 ms |

---

## 🔒 Security Notes for Production

1. Add an API key header check in `app.py`
2. Use HTTPS via Nginx reverse proxy + Let's Encrypt
3. Restrict EC2 security group port 8000 to known IPs
4. Store secrets in AWS Secrets Manager, not env variables
5. Enable CloudWatch logging for EC2
