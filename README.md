# English Proficiency Prediction — NLP

**Automatic classification of English oral proficiency from speech transcripts using Natural Language Processing and Deep Learning.**

---

## Context

This project was developed as part of the **IAS (Intelligence Artificielle et Systèmes)** module at **ENIB — École Nationale d'Ingénieurs de Brest**, during the academic period **SP9 — 2021**.

The goal is to automatically predict a speaker's English proficiency level based solely on the textual transcription of their oral production, without any human evaluation.

---

## Dataset

We used the **[NICT JLE Corpus](https://alaginrc.nict.go.jp/nict_jle/index_E.html)** (Japanese Learner English Corpus), a publicly available resource for second-language acquisition research.

| Property | Value |
|---|---|
| **Participants** | 1,281 |
| **Total words** | 1.2 million |
| **Total audio** | 300 hours |
| **Proficiency scale** | SST (Standard Speaking Test), levels **1 (low)** to **9 (high)** |
| **Data format** | Raw XML-annotated transcripts (`.txt` files) |

Each participant completed an English oral proficiency interview. The audio was transcribed and annotated with XML tags capturing linguistic phenomena such as fillers, repetitions, self-corrections, pauses, laughter, overlapping speech, and non-verbal sounds.

---

## Tasks

The project was structured around the following objectives:

1. **Preprocess the dataset**
   - Parse the raw XML transcripts from the `LearnerOriginal/` directory.
   - Extract candidate speech segments (content within `<B></B>` tags).
   - Remove all non-linguistic XML annotations (fillers, repetitions, pauses, laughter, overlapping speech, etc.).
   - Strip punctuation and normalize to lowercase.
   - Extract the SST proficiency score from the `<SST_level>` tag for each participant.
   - Write cleaned output files to an `Output/` directory.

2. **Extract features**
   - Build a Bag-of-Words (BoW) representation using `sklearn.feature_extraction.text.CountVectorizer`.
   - Construct a unified vocabulary across all 1,281 participant transcripts.

3. **Train a classifier**
   - Design a deep neural network with Keras / TensorFlow.
   - Predict the SST score (9-class classification problem).

4. **Evaluate performance**
   - Compute accuracy on a held-out validation set.
   - Generate and visualize the confusion matrix using seaborn.

5. **Improve the system (proposed extensions)**
   - Experiment with alternatives to Bag-of-Words, such as **GloVe** or other pre-trained word embeddings.
   - Incorporate dropout and regularization techniques to mitigate overfitting.

---

## Technical Approach

### Part 1 — Preprocessing (`01_data_preprocessing.ipynb`)

- **Libraries:** `re`, `os` (Python standard library only).
- **Input:** XML-annotated transcript files from Google Drive (`NICT_JLE_4.1/LearnerOriginal/`).
- **Processing:**
  - Extraction of SST score from `<SST_level>` tags.
  - Removal of undesired XML tags: `<F>`, `<R>`, `<OL>`, `<laughter>`, `<nvs>`, `<CO>`, `<H>`, `<SC>`, `<.>`, `<?>`, and any remaining `<...>` tags.
  - Removal of punctuation and special characters.
  - Lowercasing and whitespace normalization.
- **Output:** Cleaned text files saved to `NICT_JLE_4.1/Output/`, each containing the SST score on the first line followed by the cleaned transcript.

### Part 2 — Classification (`02_model_training.ipynb`)

- **Libraries:** Keras / TensorFlow, scikit-learn, NumPy, Matplotlib, seaborn.
- **Feature extraction:**
  - CountVectorizer (Bag-of-Words) builds a vocabulary of **~15,154 tokens**.
  - Each transcript is encoded as a sparse vector of token counts.
- **Model architecture** (cone-shaped feedforward network):

| Layer | Type | Units | Activation | Parameters |
|---|---|---|---|---|
| Input | — | 15,154 | — | — |
| Dense | Hidden | 2,273 | Sigmoid | 34,454,134 |
| Dense | Hidden | 341 | Sigmoid | 775,434 |
| Dense | Hidden | 51 | Sigmoid | 17,442 |
| Dense | Output | 9 | Softmax | 468 |
| **Total** | | | | **35,247,478** |

- **Training configuration:**
  - **Optimizer:** Adam
  - **Loss:** Categorical cross-entropy
  - **Epochs:** 35
  - **Batch size:** 750
  - **Train/validation split:** 80% / 20%
- **Evaluation:**
  - **Final validation accuracy:** ~61.3%
  - Confusion matrix generated for all 9 proficiency levels.

---

## Project Structure

```
English_proficiency_prediction_NLP/
├── README.md                       ← Project documentation
├── 01_data_preprocessing.ipynb     ← Part 1: Data cleaning and feature extraction
└── 02_model_training.ipynb         ← Part 2: Neural network training and evaluation
```

Both notebooks are designed to run on **Google Colab**, with the dataset mounted from Google Drive.

---

## Results

| Metric | Value |
|---|---|
| **Vocabulary size** | 15,154 tokens |
| **Model parameters** | 35,247,478 |
| **Training accuracy** | ~92.5% (epoch 35) |
| **Validation accuracy** | ~61.3% |
| **Classes** | 9 (SST levels 1–9) |

The gap between training and validation accuracy (~31 percentage points) indicates **significant overfitting**, a known limitation when working with a relatively small dataset (1,281 samples) and a high-dimensional feature space.

---

## Authors

| Name | Role |
|---|---|
| **CORREA, Elias** | Author |
| **GASSIBE, Franco** | Author |

## Supervisor

| Name | Institution |
|---|---|
| **Olivier Augereau** | ENIB |

---

## References

- [NICT JLE Corpus](https://alaginrc.nict.go.jp/nict_jle/index_E.html) — Izumi, E., Uchimoto, K., & Isahara, H. (2004). *The NICT JLE Corpus.*
- [scikit-learn: CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Keras Documentation](https://keras.io/)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
