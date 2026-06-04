# Fake News Detection with NLP

**Authors:** Lydia, Andre  
**Date:** May 2026

##  Overview
End-to-end NLP pipeline to automatically classify news articles 
as **Fake or Real** using two embedding strategies and two 
classifiers, applied to a balanced dataset of ~40,000 articles.

## Project Structure

| Section | Description |
|---------|-------------|
| **0 - Introduction** | Project goals and pipeline overview |
| **1 - EDA** | Class distribution, subject analysis, text length, word clouds |
| **2 - Feature Engineering** | Stylometric features, text cleaning, duplicate removal |
| **3 - Train/Test Split** | Stratified 80/20 split |
| **4.1 - TF-IDF Embedding** | Sparse baseline embeddings |
| **4.2 - Qwen2 Embedding** | Dense contextual embeddings (Alibaba-NLP/gte-Qwen2-1.5B) |
| **5 - Classification** | Naïve Bayes and Logistic Regression |
| **6 - Model Comparison** | Performance metrics and confusion matrices |
| **7 - Predictions** | Inference on unseen validation data |
| **8 - Conclusions** | Results summary and key learnings |

## Results Summary

| Model | Embedding | Accuracy |
|-------|-----------|----------|
| Naïve Bayes | TF-IDF | 89.6% |
| Naïve Bayes | Qwen2 | 92.8% |
| Logistic Regression | TF-IDF | ~95% |
| Logistic Regression | Qwen2  | **Best** |

**Logistic Regression + Qwen2 embeddings** selected as 
the final model for its highest accuracy and F1-score.

## Key Finding
The `subject` column is a perfect data leakage predictor 
and was excluded from all features.

## Data
The datasets are provided by Ironhack as part of the NLP 
project challenge:
- `data.csv` — main labeled dataset (~40,000 articles)
- `validation_data.csv` — unlabeled test set for final predictions

Source: [ironhack-labs/project-nlp-challenge](https://github.com/ironhack-labs/project-nlp-challenge)

## Tech Stack
- Python, pandas, numpy, scikit-learn
- HuggingFace Transformers (Qwen2)
- NLTK (POS tagging, lemmatization)
- matplotlib, seaborn, WordCloud
- PyTorch

##  How to Run
1. Open the notebook in Google Colab
2. Enable GPU: `Runtime > Change runtime type > GPU`
3. Download the data from the Ironhack source above 
   and upload to `/content/`
4. Run all cells in order
