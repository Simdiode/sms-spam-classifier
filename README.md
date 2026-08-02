# SMS Spam Classifier

A Transformer encoder built and trained **from scratch in PyTorch** (no pretrained models) to classify SMS messages as spam or ham, served as a REST API with FastAPI and Docker.

- ~97% accuracy on a held-out test set
- Small model: 2 encoder layers, 64-dim embeddings (~1.4 MB checkpoint)
- Trained on the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset (5,572 messages)

## Live demo

The API is deployed on Render:

- **Interactive docs:** https://sms-spam-classifier-wh84.onrender.com/docs — try the classifier from your browser
- **Health check:** https://sms-spam-classifier-wh84.onrender.com/health

```bash
curl -X POST https://sms-spam-classifier-wh84.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WIN a free prize now!"}'
```

> Note: the service runs on Render's free tier — the first request after a period of inactivity can take ~a minute while the instance spins up.

## Project structure

| File | Purpose |
|---|---|
| `nn.ipynb` | Full training walkthrough: data prep, vocabulary, model, training loop, evaluation |
| `notebook.ipynb` | Quick inference demo (including an interesting failure case) |
| `inference.py` | Loads the checkpoint and exposes `predict_sms(text)` |
| `app.py` | FastAPI app with `/health` and `/predict` endpoints |
| `Dockerfile` | Container image for serving the API |
| `spam_transformer.pth` | Trained checkpoint (weights + vocab + config) |
| `SMSSpamCollection` | The dataset (see `SMSSpamCollection.readme.txt` for license/citation) |

## Run the API locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (use source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
uvicorn app:app --port 8010
```

Then open http://127.0.0.1:8010/docs to try it interactively, or:

```bash
curl -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WIN a free prize now!"}'
```

Response:

```json
{"prediction": "spam", "ham_probability": 0.0007, "spam_probability": 0.9993}
```

## Run with Docker

```bash
docker build -t spam-classifier .
docker run -d -p 8010:8000 spam-classifier
```

Or pull the published image:

```bash
docker run -d -p 8010:8000 simdiode/spam-classifier:v1
```

## Retrain the model

Open `nn.ipynb` and run all cells (needs `torch`, `pandas`, `scikit-learn`). Training takes a few minutes on a modest GPU. The notebook saves the checkpoint the API serves.

## Results

```
              precision    recall  f1-score   support
         ham     0.9794    0.9848    0.9821       724
        spam     0.8981    0.8661    0.8818       112
    accuracy                         0.9689       836
```

## Dataset citation

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* ACM DOCENG'11.
