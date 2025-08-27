# Fake News Detection System 📰❌

A machine learning-based system to **detect and classify fake news articles**. Leveraging **natural language processing (NLP)** and advanced ML techniques, this system provides a reliable method to identify misinformation with high accuracy.

---

## Features

- ✅ **High-accuracy detection:** Support Vector Machine (SVM) model achieves excellent performance on fake news classification.  
- 📝 **Advanced text preprocessing:** Tokenization, lemmatization, and word embeddings using Word2Vec for robust feature extraction.  
- 💻 **Interactive interface:** Gradio-based UI for real-time article testing and user-friendly interaction.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/anaslimem/Fake-news-detection.git
cd Fake-news-detection
```
2. **Install required libraries:**

```bash
pip install numpy pandas scikit-learn nltk gensim matplotlib gradio
```

3. **Download NLTK data:**

```bash
import nltk

nltk.download('punkt')
nltk.download('wordnet')
```

---
Dataset

Source: Kaggle

Name: Fake News Detection Datasets

The dataset contains labeled news articles categorized as "Fake" or "Real".

---
## Workflow

Data Preprocessing:

   - Tokenization and lemmatization with NLTK.

   - Feature extraction using Word2Vec embeddings.

   - Model Training & Evaluation:

   - Split data into training and testing sets.

   - Train Support Vector Classifier (SVC).

   - Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.

Interactive Interface:

   - Gradio-based UI for real-time news article classification.

---

## Example Metrics

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 98%   |
| Precision | 97%   |
| Recall    | 96%   |
| F1-score  | 96%   |

---

## Libraries Used

- numpy
- pandas
- scikit-learn
- nltk
- gensim
- matplotlib
- gradio


