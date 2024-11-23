# Fake News Detection System
This project is a machine learning-based system designed to detect and classify fake news articles. By leveraging natural language processing (NLP) techniques and advanced machine learning models, the system provides a reliable method for identifying misinformation.
## Features
-High accuracy in detecting fake news using Support Vector Machines (SVM).
-Text preprocessing and word embeddings with Word2Vec for enhanced feature extraction.
-Intuitive interface built with Gradio for user interaction.
## Installation

Clone the repository:
git clone https://github.com/anaslimem/Fake-news-detection.git
cd Fake-news-detection  

Install required libraries:
pip install numpy pandas sklearn nltk gensim matplotlib gradio  

Download necessary NLTK data:
import nltk  
nltk.download('punkt')  
nltk.download('wordnet')  

##Dataset
The dataset used in this project was sourced from Kaggle. It contains labeled news articles categorized as "Fake" or "Real."

-Dataset Name: Fake News Detection Datasets
-Dataset Link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?resource=download

##Workflow
####1-Data Preprocessing:

-Tokenization and lemmatization using NLTK.
-Word embeddings generated with Word2Vec.

####2-Model Training and Testing:

-Data split into training and testing sets using train_test_split.
-Support Vector Classifier (SVC) for classification.
-Evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrix.
####3-Interactive Interface:
-A Gradio-based UI for users to input news articles and test the model's predictions in real time.

##Example Evaluation Metrics:

Accuracy: 98%
Precision: 97%
Recall: 96%
F1-score: 96%
##Dependencies
-Python 3.x
####Libraries:
-numpy
-pandas
-scikit-learn
-nltk
-gensim
-matplotlib
-gradio
##Contribution
Feel free to fork this repository and improve the project! Contributions are welcome.
##Acknowledgments
Special thanks to the creators of the libraries and tools used in this project and to Kaggle for providing the dataset.

