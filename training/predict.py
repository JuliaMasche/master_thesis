from flair.data import Corpus, Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.datasets import CSVClassificationCorpus, SentenceDataset
from word_embeddings import select_word_embedding, select_document_embeddding
from datasets import get_dataset_info
from analysis import accuracy, prec_rec_f1, conf_mat, f1, performance_measure
import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import os



parser = argparse.ArgumentParser()
parser.add_argument("--path", type = str)
args = parser.parse_args()


def create_text_label_list(path:str):
    df = pd.read_csv(path, sep = '\t')
    text = df['sentence'].tolist()
    text = [str(i).lower() for i in text]
    stop = set(stopwords.words('german'))
    from nltk.tokenize import word_tokenize
    X = []
    for x in text:
        tokens = word_tokenize(x)
        result = [i for i in tokens if not i in stop]
        x = (" ").join(result)
        X.append(x)
    X = np.asanyarray(X)
    labels = df['label'].tolist()
    labels = [str(i) for i in labels]
    return X, np.asarray(labels)

def predict(test_text, classifier):
    test_pred = []
    for item in test_text:
        sentence = Sentence(item)
        classifier.predict(sentence)
        result = sentence.labels
        test_pred.append(result[0].value)
    return test_pred

train_file = os.path.join(args.path, 'train.tsv')
train_text, train_labels = create_text_label_list(train_file)
classes = set(train_labels)
classes = list(classes)
print(classes)

classifier = TextClassifier.load('../results/Scan/Transformer/bert/no_al_training/resources/training/final-model.pt')
pred_label = predict(train_text, classifier)
score = performance_measure(train_labels, pred_label, 'weighted' , 'g-mean', [], classes)
conf_mat(train_labels, pred_label, classes, '../results/Scan/Transformer/bert/no_al_training/')
print(score)
