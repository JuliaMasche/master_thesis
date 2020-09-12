from flair.data import Corpus, Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.datasets import CSVClassificationCorpus, SentenceDataset
from word_embeddings import select_word_embedding, select_document_embeddding
from datasets import get_dataset_info
from analysis import accuracy, prec_rec_f1, conf_mat, f1, performance_measure
from seed_set import select_random
from query_strategy import select_query_strategy, select_next_batch
import csv
import copy
import os
import json
import repackage
repackage.up()
from alipy import ToolBox
import pandas as pd
import numpy as np
import argparse
import timeit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
import shutil
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from torch.optim.adam import Adam

query_strategy = ["QueryInstanceUncertainty", "QueryInstanceRandom", "QueryInstanceGraphDensity", "QueryInstanceQBC", "QueryExpectedErrorReduction", "QueryInstanceLAL_indep", "QueryInstanceLAL_iter"]
we_embeddings = ['glove', 'flair', 'fasttext', 'bert', 'word2vec', 'elmo_small', 'elmo_medium', 'elmo_original']
doc_embeddings = ["Pool", "RNN", "Transformer_eng", "Transformer_ger"]
sets = ["SST-2_original", "SST-2_90", "SST-2_80", "SST-2_70", "SST-2_60", "SST-2_50", "wiki_1000", "wiki_3000", "wiki_1000_wo_unknown", "webkb_1000", "webkb_2000", "movie_70", "movie_80", "news_1000", "news_2000", "news_3000", "news_balanced", "scan"]
column_name_map = {0: 'text', 1: 'label'}


parser = argparse.ArgumentParser()
parser.add_argument("-stop", "--stopping_criterion", default = 10, type = int)
parser.add_argument("--dataset", choices=sets, default = "SST-2", type = str)
parser.add_argument("-qs", "--query_str", choices=query_strategy, default = "QueryInstanceUncertainty", type = str)
parser.add_argument("-we", "--word_embedding", choices=we_embeddings, default = "glove", type = str)
parser.add_argument("-de", "--document_embedding", choices=doc_embeddings, default = "Pool", type = str)
parser.add_argument("-lr", "--learning_rate", default = 0.01, type = float)
parser.add_argument("-mini_b", "--mini_batch_size", default = 32, type = int)
parser.add_argument("-ep", "--max_epoch", default = 100, type = int)
parser.add_argument("--seed", default = 5, type = int)
parser.add_argument("-bs", "--batch_size", default = 20, type = int)
parser.add_argument("-pm", "--perf_measure", default = "accuracy", type = str)
parser.add_argument("-num_cl", "--num_classifier", default = 2, type = int)
args = parser.parse_args()
dataset = args.dataset
seed = args.seed
dataset_path_original, out_dir, classes, sep, minority, average = get_dataset_info(dataset)
stopping_crit = args.stopping_criterion
query_str = args.query_str
word_embedding = args.word_embedding
document_embedding = args.document_embedding
mini_batch_size = args.mini_batch_size
max_epoch = args.max_epoch

path_tmp = document_embedding + '/' + word_embedding + "/" + query_str 
path_results = os.path.join(out_dir, path_tmp)
try:
    os.makedirs(path_results, exist_ok = True)
except OSError:
    print ("Creation of the directory %s failed" % path_results)
else:
    print ("Successfully created the directory %s " % path_results)



def update_train_set(label_ind, df_train_updated, text, labels, path):
    #not used anymore
    for idx in label_ind:
        df_train_updated = df_train_updated.append({'text':text[idx],'labels': labels[idx], 'original_idx': idx}, ignore_index=True)
    df_train_updated.to_csv(os.path.join(path, 'train.tsv'), index=False, sep = sep)


def create_feat_mat(text, document_embeddings):
    X = []
    for txt in text:
        sentence = Sentence(txt)
        document_embeddings.embed(sentence)
        embedding = sentence.get_embedding().cpu().detach().numpy()
        X.append(embedding)
    return X


def create_pred_mat(unlab_ind, classifier, train_text):
    pred_ma = []
    for idx in unlab_ind:
        p = []
        sentence = Sentence(train_text[idx])
        classifier.predict(sentence, multi_class_prob=True)
        result = sentence.labels
        for r in result:
            p.append(r.score)
        pred_ma.append(p)
    return pred_ma


def create_pred_mat_one(unlab_ind, classifier, train_text):
    pred_ma = []
    for idx in unlab_ind:
        sentence = Sentence(train_text[idx])
        classifier.predict(sentence)
        result = sentence.labels
        pred_ma.append(result[0].score)
    return pred_ma


def create_pred_mat_class(unlab_ind, classifier, train_text):
    pred_ma = []
    for idx in unlab_ind:
        sentence = Sentence(train_text[idx])
        classifier.predict(sentence)
        result = sentence.labels
        pred_ma.append(result[0].value)
    print(pred_ma)
    return pred_ma


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
    if args.document_embedding == "Transformer_ger":
        stop = set(stopwords.words('german'))
    else:
        stop = set(stopwords.words('english'))
    from nltk.tokenize import word_tokenize
    X = []
    idx = []
    for x in range(len(df)):
        text = df['sentence'][x]
        tokens = word_tokenize(text)
        result = [i for i in tokens if not i in stop]
        if not result:
            idx.append(x)
        new = (" ").join(result)
        df['sentence'][x] = new.lower()
    df = df.drop(index = idx)
    X = df['sentence'].tolist()
    labels = df['label'].tolist()
    labels = [str(i) for i in labels]
    return np.asarray(X), np.asarray(labels)



def create_sentence_dataset(train_text, train_labels):
    datapoints = []
    for i in range(len(train_text)):
        label = str(train_labels[i])
        datapoints.append(Sentence(train_text[i]).set_label('label',label))
    return datapoints


def write_json(dictionary, path, size, arg):
    json_object = json.dumps(dictionary, indent = size) 
  
    with open(os.path.join(path, "config.json"), arg) as outfile: 
        outfile.write(json_object) 


def train_trainer(document_embeddings, label_dict, corpus, learning_rate, path):
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
    
    if args.document_embedding == "Pool" or args.document_embedding == "RNN":
        trainer = ModelTrainer(classifier, corpus)

        trainer.train(os.path.join(path_results, path),
                    learning_rate=learning_rate,
                    mini_batch_size=mini_batch_size,
                    anneal_factor=0.5,
                    patience=5,
                    max_epochs=max_epoch,
                    train_with_dev = True,
                    embeddings_storage_mode= "gpu")

    elif args.document_embedding == "Transformer_eng" or args.document_embedding == "Transformer_ger":
        trainer = ModelTrainer(classifier, corpus, optimizer= Adam)

        trainer.train(os.path.join(path_results, path),
                    learning_rate=learning_rate, 
                    mini_batch_size=mini_batch_size,
                    mini_batch_chunk_size=4, 
                    max_epochs=5,
                    embeddings_storage_mode= "gpu")


def al_main_loop(alibox, al_strategy, document_embeddings, train_text, train_labels, test_text, test_labels, datapoints, label_ind, unlab_ind, y):
    shutil.rmtree(os.path.join(path_results, 'resources/training'), ignore_errors=True)
    num_instances = []
    performance = []
    num_queries = 0

    starttime = timeit.default_timer()

    #first training without querying

    #create train sentences
    train_sentences = []
    for idx in label_ind:
        train_sentences.append(datapoints[idx])
    
    #initialize trainer
    train = SentenceDataset(train_sentences)
    corpus = Corpus(train=train)
    label_dict = corpus.make_label_dictionary()
    #train
    if args.document_embedding == "Pool" or args.document_embedding == "RNN":
        train_trainer(document_embeddings, label_dict, corpus, args.learning_rate, 'resources/training')
    elif args.document_embedding == "Transformer_eng" or args.document_embedding == "Transformer_ger":
        train_trainer(document_embeddings, label_dict, corpus, 3e-5, 'resources/training')
    #load classifier and create pred test set
    classifier = TextClassifier.load(os.path.join(path_results, 'resources/training/final-model.pt'))
    test_pred = predict_testset(test_text, classifier)

    #if DocumentRNNEmbeddings, then update feature matrix of AL with trained embeddings
    if args.document_embedding == "RNN" or args.document_embedding == "Transformer_eng" or args.document_embedding == "Transformer_ger":
        document_embeddings_trained = classifier.document_embeddings
        feat_mat = create_feat_mat(train_text, document_embeddings_trained)
        idx = list(range(0, len(train_text)))
        alibox = ToolBox(X=feat_mat, y=y, query_type='AllLabels', saving_path='.')
        al_strategy = select_query_strategy(alibox, query_str, idx)

    #calculate score
    score = performance_measure(test_labels, test_pred, average, args.perf_measure, minority, classes)
    num_instances.append(len(label_ind))
    performance.append(score)


    while len(unlab_ind) != 0:
        num_queries += 1

        #if QBC strategy, train second committte member and create two prediction matrices; for Random none; else one predicition matrix
        if query_str == "QueryInstanceQBC":
            pred_mat = []
            pred_mat.append(create_pred_mat_class(unlab_ind, classifier, train_text))
            word_embeddings = select_word_embedding(word_embedding)
            document_embeddings = select_document_embeddding(document_embedding, word_embeddings)
            if args.document_embedding == "Pool" or args.document_embedding == "RNN":
                learning_rate = args.learning_rate + 0.05
            elif args.document_embedding == "Transformer_eng" or args.document_embedding == "Transformer_ger":
                learning_rate = 3e-4
            train_trainer(document_embeddings, label_dict, corpus, learning_rate, 'resources/QBC/training')
            classifier_two = TextClassifier.load(os.path.join(path_results, 'resources/QBC/training/final-model.pt'))
            pred_mat.append(create_pred_mat_class(unlab_ind, classifier_two, train_text))
            shutil.rmtree(os.path.join(path_results, 'resources/QBC/training'), ignore_errors=True)
        elif query_str == "QueryInstanceRandom":
            pred_mat = []
        else:
            pred_mat = create_pred_mat(unlab_ind, classifier, train_text)

        #empty cache
        torch.cuda.empty_cache()
        shutil.rmtree(os.path.join(path_results, 'resources/training'), ignore_errors=True)

        #initialize word/document embeddings new after every query
        word_embeddings = select_word_embedding(word_embedding)
        document_embeddings = select_document_embeddding(document_embedding, word_embeddings)

        #select new labeled instances 
        if len(unlab_ind) < args.batch_size:
            select_ind = select_next_batch(al_strategy, query_str, label_ind, unlab_ind, len(unlab_ind), pred_mat)
        else:
            select_ind = select_next_batch(al_strategy, query_str, label_ind, unlab_ind, args.batch_size, pred_mat)
        label_ind.extend(select_ind)
        unlab_ind = [n for n in unlab_ind if n not in select_ind]

        #update train sentences with new labeled data
        train_sentences = []
        for idx in label_ind:
            train_sentences.append(datapoints[idx])
        
        #train new query
        train = SentenceDataset(train_sentences)
        corpus = Corpus(train=train)
        label_dict = corpus.make_label_dictionary()
        if args.document_embedding == "Pool" or args.document_embedding == "RNN":
            train_trainer(document_embeddings, label_dict, corpus, args.learning_rate, 'resources/training')
        elif args.document_embedding == "Transformer_eng" or args.document_embedding == "Transformer_ger":
            train_trainer(document_embeddings, label_dict, corpus, 3e-5, 'resources/training')
        classifier = TextClassifier.load(os.path.join(path_results, 'resources/training/final-model.pt'))
        test_pred = predict_testset(test_text, classifier)

        if args.document_embedding == "RNN" or args.document_embedding == "Transformer_ger" or args.document_embedding == "Transformer_eng":
            document_embeddings_trained = classifier.document_embeddings
            feat_mat = create_feat_mat(train_text, document_embeddings_trained)
            idx = list(range(0, len(train_text)))
            alibox = ToolBox(X=feat_mat, y=y, query_type='AllLabels', saving_path='.')
            al_strategy = select_query_strategy(alibox, query_str, idx)

        score = performance_measure(test_labels, test_pred, average, args.perf_measure, minority, classes)
        num_instances.append(len(label_ind))
        performance.append(score)

    #runtime, report, update json
    runtime = timeit.default_timer() - starttime
    run_dict = {"runtime" :runtime}
    write_json(run_dict, path_results, len(run_dict), "a")
    report = prec_rec_f1(test_labels, test_pred, classes)
    write_json(report, path_results, len(report), "a")
    dict_perf = dict(zip(num_instances, performance))
    write_json(dict_perf, path_results, len(dict_perf), "a")

    return num_instances, performance, label_ind, runtime
    

def main_func():

    #create config file
    config = {
        "query_strategy": query_str,
        "word_embedding": word_embedding,
        "document_embedding": document_embedding,
        "neural network": "rnn",
        "dataset": dataset_path_original,
        "learning_rate": args.learning_rate,
        "max epoch": max_epoch,
        "mini batch size": mini_batch_size,
        "stopping_criterion" : stopping_crit,
        "seed" : args.seed,
        "query batch size" : args.batch_size,
        "performance_measure": args.perf_measure,
        "num_classifier_QBC": args.num_classifier
        }

    write_json(config, path_results, len(config), "w")

    #load file and create lists
    train_file = os.path.join(dataset_path_original, 'train.tsv')
    train_text, train_labels = create_text_label_list(train_file)
    le = preprocessing.LabelEncoder()
    le.fit(classes)

    #initialze KFold
    cv = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
    kfold_label_idx = []
    test_idx_list = []
    overall_perf = {}
    overall_run = []
    seed_label = 5
    for train_index, test_index in cv.split(train_text, train_labels):
        test_idx_list.append(test_index)
        X_train, X_test = train_text[train_index], train_text[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        datapoints = create_sentence_dataset(X_train, y_train)
        y = le.transform(y_train)
        
        #initialze embeddings
        word_embeddings = select_word_embedding(word_embedding)
        document_embeddings = select_document_embeddding(document_embedding, word_embeddings)
        
        #create feature matrix
        feat_mat = create_feat_mat(X_train, document_embeddings)
        idx = list(range(0, len(X_train)))

        dict_train_to_train_idx = dict(zip(idx, train_index))
        dict_instances = {"train_length": len(X_train), "test_length": len(X_test)}
        write_json(dict_instances, path_results, len(dict_instances), "a")

        #initialize Active Learning
        alibox = ToolBox(X=feat_mat, y=y, query_type='AllLabels', saving_path='.')
       
        #select random label and unlab set
        label_ind, unlab_ind = select_random(seed_label, idx, 100)
        seed_label = seed_label + 1


        seed_set_labels = list(y_train[label_ind])
        dict_seedset = dict(zip(label_ind, seed_set_labels))
        write_json(dict_seedset, path_results, len(dict_seedset), "a")

        #select al strategy
        al_strategy = select_query_strategy(alibox, query_str, idx)

        #start querying
        num_instances, performance, label_ind_new, runtime = al_main_loop(alibox, al_strategy, document_embeddings, X_train, y_train, X_test, y_test, datapoints, label_ind, unlab_ind, y)
        overall_run.append(runtime)
        original_kfold_idx = []
        for i in range(len(label_ind_new)):
            original_kfold_idx.append(dict_train_to_train_idx[label_ind_new[i]])
        kfold_label_idx.append(original_kfold_idx)
        

        for i in range(len(num_instances)):
            key = str(num_instances[i])
            if key in overall_perf:
                overall_perf[key] = overall_perf[key] + performance[i]
            else:
                overall_perf[key] = performance[i]


    #final analysis
    for i in overall_perf:
        overall_perf[i] = overall_perf[i]/5
    df_perform = pd.DataFrame(overall_perf.items(), columns=['num_instances','performance'])
    df_perform.to_csv(os.path.join(path_results, "df_perform.tsv"), sep = '\t')

    sum_run = sum(overall_run)
    dict_all_runtimes = {"overall_runtime": sum_run/5}
    write_json(dict_all_runtimes, path_results, len(dict_all_runtimes), "a")

    label_dict = {}
    for i in range(len(kfold_label_idx)):
        name = "kfold" + str(i+1)
        label_dict[name] = list(kfold_label_idx[i])
    df_label_idx = pd.DataFrame(label_dict)
    df_label_idx.to_csv(os.path.join(path_results, "df_label_idx_per_kfold.tsv"), sep = '\t')

    test_dict = {}
    for i in range(len(test_idx_list)):
        name = "kfold" + str(i+1)
        test_dict[name] = list(test_idx_list[i])
    df_test_idx = pd.DataFrame(test_dict)
    df_test_idx.to_csv(os.path.join(path_results, "df_test_idx.tsv"), sep = '\t')

main_func()
        

