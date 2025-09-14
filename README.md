**Duke ECE Data Science – Emotion Classification (Final Project)**
- **Goal:** Multi-class emotion classification on short texts (6 emotions: anger, fear, joy, love, sadness, surprise).
- **Highlights:** Clean, modular Python pipeline; reproducible baselines; optional BERT fine-tuning; clear metrics and training curves.
- **Provenance:** This repository is a professional re-build of our Duke ECE Data Science final group project. The original files are archived under `scratch/`, and the final report PDF is available at `docs/final_report.pdf`.

**Project Structure**
- `src/emotion_nlp`: Core library with data loading, preprocessing, models, training, and evaluation.
- `data`: Raw class files (`*_1000_clean.txt`) used to construct splits.
- `assets`: Figures used in the README; generated plots go in `assets/generated/`.
- `artifacts`: Auto-created during runs for saved splits and metrics.
- `scratch`: Archive of the original project files (not used by the new pipeline).

**Results & Figures**
- Below are representative training curves and figures from our experiments, aligned with the final report.
  - RandomForest training curve (accuracy vs. `n_estimators`):
    - `assets/training_curve_1.png`
    - `assets/training_curve_2.png`
  - Additional report figures:
    - `assets/training_curve_3.png`
    - `assets/training_curve_4.png`
    - `assets/training_curve_5.png`

To reproduce a similar curve with the modern pipeline, run the RF trainer; it will also export `assets/generated/training_curve_rf.png`.

**Quickstart**
- Create and activate a Python 3.10+ environment.
- Install requirements: `pip install -r requirements.txt`
- Ensure raw data files exist in `data/`:
  - `anger_1000_clean.txt`, `fear_1000_clean.txt`, `joy_1000_clean.txt`, `love_1000_clean.txt`, `sad_1000_clean.txt`, `surprise_1000_clean.txt`

Run baseline (RandomForest)
- `python -m src.emotion_nlp.train --data-dir data`
- Outputs:
  - Train/val/test splits in `artifacts/data/`
  - Metrics in `artifacts/rf_metrics.txt`
  - Training curve in `assets/generated/training_curve_rf.png`

**Data Acquisition**
- Source: We collected real tweets from X (formerly Twitter) using a developer account and the Tweepy library. For each emotion category (joy, love, fear, anger, surprise, sadness), we queried user-authored content associated with that tag and retrieved candidates, then cleaned for language and quality. From this pool, we sampled 100 high-quality originals per class after removing non-English text, special characters, and incomplete sentences, followed by a final human verification pass.
- Augmentation: To scale the dataset, we generated 900 additional sentences per class using ChatGPT-4o (total 1,000 per class). The prompt enforced diversity and clarity of emotional cues: “Generate 100 emotion-labeled sentences under the [category]. Each sentence should be 1–2 sentences long and include clear emotional cues through concrete details and imagery. The tone should be natural and grounded in daily experiences. [Strict] Do not repeat with the previous generated sentences.” We repeated this 9 times per class. The final dataset thus comprises balanced class files used by the pipeline.

Notes
- Data cleaning included lowercasing, removal of URLs/special characters, and filtering of non-English inputs. We also handled edge cases in the raw files where stray semicolons could break parsing, preserving the intended text and label.
- See `docs/final_report.pdf` for the full methodology and analysis details.

Optional: BERT Fine-Tuning (DistilBERT)
- The repository includes a minimal wrapper to initialize DistilBERT (see `src/emotion_nlp/models.py`).
- For full fine-tuning, we recommend a GPU environment; adapt a trainer script using the provided tokenizer/model (e.g., PyTorch or Hugging Face `Trainer`).

**Library Overview**
- `src/emotion_nlp/data.py`:
  - Reads the six class files, applies the same semicolon-cleaning logic as our original code, shuffles, and splits into train/val/test.
  - `build_splits(...)` to create splits; `save_splits(...)` to persist CSVs.
- `src/emotion_nlp/preprocess.py`:
  - `TextProcessor` transformer mirrors our original preprocessing (case-folding, non-alpha removal, optional stemming, stopword removal).
- `src/emotion_nlp/models.py`:
  - `build_rf_pipeline(...)` returns a scikit-learn pipeline with `TextProcessor`, `CountVectorizer`, and `RandomForestClassifier`.
  - Optional `build_distilbert(...)` helper constructs tokenizer/model for DistilBERT.
- `src/emotion_nlp/evaluate.py`:
  - Accuracy, confusion matrix, classification report; macro/micro AUC when probabilities are available.
- `src/emotion_nlp/train.py`:
  - CLI entry for the RF baseline; builds splits, trains, evaluates, logs metrics, and saves a training curve.

**Reproducibility Notes**
- Data handling and cleaning logic follow our final report and original notebooks/scripts, including nuanced semicolon fixes in the raw text files.
- Metrics are reported consistently across splits; AUC is computed in a one-vs-rest fashion (macro/micro).
- NLTK stopwords are used by default; on first run, NLTK may download data (internet required for that step). If offline, change the `TextProcessor` to skip stopwords or pre-bundle the corpus.

**Extending the Project**
- Swap vectorizers (e.g., `TfidfVectorizer`) and classifiers (e.g., Linear SVM, Logistic Regression) within the same pipeline.
- Integrate Hugging Face `Trainer` for end-to-end DistilBERT fine-tuning with logging, checkpoints, and LR scheduling.
- Add cross-validation and hyperparameter sweeps (e.g., `sklearn.model_selection.GridSearchCV`).

**Team & Acknowledgements**
- This is the final group project for Duke University’s ECE Data Science course. Many thanks to course staff and collaborators.
