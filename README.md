# Social Media Sentiment Analysis System

A fully-featured Python pipeline that ingests social-media text, cleans it, predicts sentiment with rule-based and machine-learning models, and produces rich visualisations & reports.

---

## ✨ Key Features

* End-to-end pipeline: **load → preprocess → model → visualise → report**
* Hybrid sentiment engine combining **VADER**, **TextBlob** & a **Logistic-Regression TF-IDF** model (scikit-learn).
* Beautiful static plots (Matplotlib/Seaborn) and interactive dashboards (Plotly).
* Automatic HTML & Excel reports with summary statistics, feature analysis and embedded graphics.
* Modular architecture – each stage lives in its own package (preprocessing, modeling, visualisation, utils).

---

## 🗂️ Project Structure

```
├── data/                     # Raw or prepared CSV data
├── notebooks/                # Jupyter exploration
├── models/                   # Saved ML models
│   └── sentiment_model/
├── reports/                  # Auto-generated reports & figures
├── src/                      # Source code package
│   ├── preprocessing/
│   │   └── text_processor.py
│   ├── modeling/
│   │   └── sentiment_model.py
│   ├── visualization/
│   │   └── visualizer.py
│   ├── utils/
│   │   └── report_generator.py
│   └── main.py               # Pipeline entry-point
└── README.md                 # (you are here)
```

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd Social_Media_Sentiment_System
   ```
2. **Create & activate a virtualenv (recommended)**
3. **Install dependencies**
   ```bash
   # Requirements file not committed yet – install manually:
   pip install pandas numpy nltk textblob vaderSentiment matplotlib seaborn plotly wordcloud scikit-learn joblib openpyxl
   ```
   After installation, download the small NLTK corpora the first time you run the code; the `TextPreprocessor` downloads them automatically if missing.

> 📦 Feel free to generate a `requirements.txt` with `pip freeze > requirements.txt` for reproducible installs.

---

## 🚀 Quickstart

1. Place a CSV file in `data/` (default name: `social_media_data.csv`).  Required columns:
   * `text` – the post/tweet text (string)
   * `date` – ISO date string or timestamp (optional but enables trend plots)
   * Any additional columns (e.g. `category`) are preserved for analysis.

2. Run the full pipeline:
   ```bash
   python -m src.main            # or  python src/main.py
   ```

3. Outputs:
   * `reports/sentiment_analysis_results.csv` – augmented dataset containing:
     * `processed_text`, `tokens`, **`predicted_sentiment`**, probability scores, etc.
   * PNG/HTML visualisations per plot.
   * Timestamped HTML and Excel reports inside `reports/`.

---

## 📊 Interpreting Results

| Value | Meaning |
|-------|---------|
| **1** | Positive sentiment |
| **0** | Neutral sentiment |
| **-1** | Negative sentiment |

The `predicted_sentiment` column (note: renamed from older `sentiment`) carries these labels.  Visualisations and reports rely on this name.

---

## 🛠️ Extending / Customising

* **Model** – Swap in any scikit-learn classifier inside `src/modeling/sentiment_model.py`.
* **Visuals** – Add new plots in `src/visualization/visualizer.py` then embed them in reports.
* **Data** – Support more languages by replacing the preprocessing steps or using language-specific sentiment models.

---
