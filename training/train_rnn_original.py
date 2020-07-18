from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.visual.training_curves import Plotter
from flair.datasets import CSVClassificationCorpus, SentenceDataset
from datasets import get_dataset_info
from word_embeddings import select_word_embedding, select_document_embeddding
import flair
import argparse
import os
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import json
import timeit
from analysis import accuracy, prec_rec_f1, conf_mat, f1, performance_measure
import shutil
import psutil 
from sklearn import preprocessing
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')


we_embeddings = ['glove', 'flair', 'fasttext', 'bert', 'word2vec', 'elmo_small', 'elmo_medium', 'elmo_original']
doc_embeddings = ["Pool", "RNN", "Transformer"]
sets = ["SST-2_original", "SST-2_90", "SST-2_80", "SST-2_70", "SST-2_60", "SST-2_50", "wiki_1000", "webkb_1000", "webkb_2000", "news_1000", "news_2000", "news_1500", "movie_80"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=sets, default = "SST-2_50", type = str)
parser.add_argument("-we", "--word_embedding", choices=we_embeddings, default = "glove", type = str)
parser.add_argument("-de", "--document_embedding", choices=doc_embeddings, default = "Pool", type = str)
parser.add_argument("-lr", "--learning_rate", default = 0.01, type = float)
parser.add_argument("-mini_b", "--mini_batch_size", default = 32, type = int)
parser.add_argument("-ep", "--max_epoch", default = 100, type = int)
parser.add_argument("--seed", default = 5, type = int)
parser.add_argument("-pm", "--perf_measure", default = "accuracy", type = str)
args = parser.parse_args()
dataset = args.dataset
dataset_path_original, out_dir, classes, sep, minority, average = get_dataset_info(dataset)
word_embedding = args.word_embedding
document_embedding = args.document_embedding
learning_rate = args.learning_rate
mini_batch_size = args.mini_batch_size
max_epoch = args.max_epoch

tmp = word_embedding + "/no_al_training"
path_results = os.path.join(out_dir, tmp)
try:
    os.makedirs(path_results, exist_ok = True)
except OSError:
    print ("Creation of the directory %s failed" % path_results)
else:
    print ("Successfully created the directory %s " % path_results)

def write_json(dictionary, path, size, arg):
    json_object = json.dumps(dictionary, indent = size) 
  
    with open(os.path.join(path, "config.json"), arg) as outfile: 
        outfile.write(json_object) 

def predict_testset(test_text, classifier):
    test_pred = []
    for item in test_text:
        sentence = Sentence(item)
        classifier.predict(sentence)
        result = sentence.labels
        test_pred.append(result[0].value)
    return test_pred


def create_text_label_list(path:str):
    df = pd.read_csv(path, sep = sep)
    text = df['sentence'].tolist()
    text = [str(i).lower() for i in text]
    stop = set(stopwords.words('english'))
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


def create_sentence_dataset(train_text, train_labels):
    datapoints = []
    for i in range(len(train_text)):
        label = str(train_labels[i])
        datapoints.append(Sentence(train_text[i]).set_label('label',label))
    return datapoints


def main_train(datapoints, test_text, test_labels, document_embeddings):
    starttime = timeit.default_timer()

    train = SentenceDataset(datapoints)
    corpus = Corpus(train=train)
    label_dict = corpus.make_label_dictionary()

    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

    if args.document_embedding == "Pool" or args.document_embedding == "RNN":
        trainer = ModelTrainer(classifier, corpus)

        trainer.train(os.path.join(path_results,'resources/training'),
                    learning_rate= learning_rate,
                    mini_batch_size=mini_batch_size,
                    anneal_factor=0.5,
                    patience=5,
                    max_epochs=max_epoch,
                    train_with_dev = True,
                    num_workers = 6,
                    embeddings_storage_mode ="gpu")

    elif args.document_embedding == "Transformer":
        trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

        trainer.train(os.path.join(path_results, 'resources/training'),
                    learning_rate= 3e-5, # use very small learning rate
                    mini_batch_size=mini_batch_size,
                    mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
                    max_epochs=5, # terminate after 5 epochs
                    embeddings_storage_mode= "gpu")
                    
    

    classifier = TextClassifier.load(os.path.join(path_results, 'resources/training/final-model.pt'))
    test_pred = predict_testset(test_text, classifier)

    #acc = accuracy(test_labels, test_pred)
    score = performance_measure(test_labels, test_pred, average, args.perf_measure, minority, classes)
    torch.cuda.empty_cache()
    report = prec_rec_f1(test_labels, test_pred, classes)
    write_json(report, path_results, len(report), "a")
    runtime = timeit.default_timer() - starttime
    run_dict = {"runtime" : runtime}
    write_json(run_dict, path_results, len(run_dict), "a")
    shutil.rmtree(os.path.join(path_results, 'resources/training'), ignore_errors=True)
    del classifier, test_pred
    return score, runtime


def main():

    config = {
        "word_embedding": word_embedding,
        "document_embedding": document_embedding,
        "neural network": "rnn",
        "dataset": dataset_path_original,
        "learning_rate": args.learning_rate,
        "max epoch": max_epoch,
        "mini batch size": mini_batch_size,
        "seed" : args.seed,
        "performance_measure": args.perf_measure
        }

    write_json(config, path_results, len(config), "w")

    train_file = os.path.join(dataset_path_original, 'train.tsv')
    train_text, train_labels = create_text_label_list(train_file)
    #le = preprocessing.LabelEncoder()
    #le.fit(classes)

    performance = []
    runtimes = []
    overall_perf = 0
    overall_run = 0
    test_idx_list = []
    fold = list(range(1,11))
    cv = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
    for train_index, test_index in cv.split(train_text, train_labels):
        test_idx_list.append(test_index)
        X_train, X_test = train_text[train_index], train_text[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        #y_train = le.transform(y_original_train)
        #y_test = le.transform(y_original_test)
        datapoints = create_sentence_dataset(X_train, y_train)

        word_embeddings = select_word_embedding(word_embedding)
        document_embeddings = select_document_embeddding(document_embedding, word_embeddings)
        
        score, runtime = main_train(datapoints, X_test, y_test, document_embeddings)
        performance.append(score)
        runtimes.append(runtime)
    
    dict_perf = dict(zip(fold, performance))
    write_json(dict_perf, path_results, len(dict_perf), "a")

    for i in range(len(performance)):
        overall_perf = overall_perf + performance[i]
        overall_run = overall_run + runtimes[i]

    dict_all_perf = {"overall_perf" : overall_perf/5}
    dict_all_runtimes = {"overall_runtime": overall_run/5}
    write_json(dict_all_perf, path_results, len(dict_all_perf), "a")
    write_json(dict_all_runtimes, path_results, len(dict_all_runtimes), "a")

    test_dict = {}
    for i in range(len(test_idx_list)):
        name = "kfold" + str(i+1)
        test_dict[name] = list(test_idx_list[i])
    df_test_idx = pd.DataFrame(test_dict)
    df_test_idx.to_csv(os.path.join(path_results, "df_test_idx.tsv"), sep = '\t')


main()

       