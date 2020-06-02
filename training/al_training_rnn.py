from flair.data import Corpus, Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.datasets import CSVClassificationCorpus, SentenceDataset
from word_embeddings import select_word_embedding
from datasets import get_dataset_info
from analysis import accuracy, prec_rec_f1, conf_mat, f1
from seed_set import select_random
from query_strategy import select_query_strategy, select_next_batch
import csv
import copy
import os
import json
from alipy import ToolBox
import pandas as pd
import numpy as np
import argparse
import timeit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing


query_strategy = ["QueryInstanceUncertainty", "QueryInstanceRandom", "QueryInstanceGraphDensity", "QueryInstanceBMDR", "QueryInstanceSPAL", "QueryInstanceQBC"]
we_embeddings = ['glove', 'flair', 'fasttext', 'bert', 'word2vec', 'elmo_small', 'elmo_medium', 'elmo_original']
sets = ["SST-2_90", "SST-2_80", "SST-2_70", "SST-2_60", "SST-2_50", "news_1", "news_2", "news_3", "news_4", "webkb", "movie_60", "movie_80"]
column_name_map = {0: 'text', 1: 'label'}


parser = argparse.ArgumentParser()
parser.add_argument("-stop", "--stopping_criterion", default = 10, type = int)
parser.add_argument("--dataset", choices=sets, default = "SST-2", type = str)
parser.add_argument("-qs", "--query_str", choices=query_strategy, default = "QueryInstanceUncertainty", type = str)
parser.add_argument("-we", "--word_embedding", choices=we_embeddings, default = "glove", type = str)
parser.add_argument("-lr", "--learning_rate", default = 0.01, type = float)
parser.add_argument("-mini_b", "--mini_batch_size", default = 32, type = int)
parser.add_argument("-ep", "--max_epoch", default = 100, type = int)
parser.add_argument("--seed", default = 5, type = int)
parser.add_argument("-bs", "--batch_size", default = 20, type = int)
args = parser.parse_args()
dataset = args.dataset
seed = args.seed
dataset_path_original, out_dir, classes, sep, minority, average = get_dataset_info(dataset)
stopping_crit = args.stopping_criterion
query_str = args.query_str
word_embedding = args.word_embedding
mini_batch_size = args.mini_batch_size
max_epoch = args.max_epoch

path_tmp = word_embedding + "/" + query_str 
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
        sentence = Sentence(train_text[idx])
        classifier.predict(sentence)
        result = sentence.labels
        if result[0].value is 1:
            pred_ma.append([result[0].score, (1-result[0].score)])
        else:
            pred_ma.append([(1-result[0].score), result[0].score])
    return pred_ma


def create_pred_mat_class(unlab_ind, classifier, train_text):
    pred_ma = []
    for idx in unlab_ind:
        sentence = Sentence(train_text[idx])
        classifier.predict(sentence)
        result = sentence.labels
        pred_ma.append(result[0].value)
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
    text = np.asarray(df['sentence'].tolist())
    labels = df['label'].tolist()
    labels = [str(i) for i in labels]
    return text, np.asarray(labels)


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
    
    trainer = ModelTrainer(classifier, corpus)

    trainer.train(os.path.join(path_results, path),
                learning_rate=learning_rate,
                mini_batch_size=mini_batch_size,
                anneal_factor=0.5,
                patience=5,
                max_epochs=max_epoch,
                embeddings_storage_mode= "gpu")


def al_main_loop(alibox, al_strategy, document_embeddings, train_text, train_labels, test_text, test_labels, datapoints, label_ind, unlab_ind):
    
    num_instances = []
    performance = []
    num_queries = 0

    starttime = timeit.default_timer()

    #first training without querying
    train_sentences = []
    for idx in label_ind:
        train_sentences.append(datapoints[idx])
        
    train = SentenceDataset(train_sentences)
    corpus = Corpus(train=train)
    label_dict = corpus.make_label_dictionary()
    train_trainer(document_embeddings, label_dict, corpus, args.learning_rate, 'resources/training')
    classifier = TextClassifier.load(os.path.join(path_results, 'resources/training/final-model.pt'))
    pred_mat = create_pred_mat(unlab_ind, classifier, train_text)
    test_pred = predict_testset(test_text, classifier)

    f1_score = f1(test_labels, test_pred, average, minority)
    num_instances.append(len(label_ind))
    performance.append(f1_score)


    while num_queries < stopping_crit:
        num_queries += 1

        if query_str == "QueryInstanceQBC":
            pred_mat = []
            pred_mat.append(create_pred_mat_class(unlab_ind, classifier, train_text))
            learning_rate = args.learning_rate + 0.05
            train_trainer(document_embeddings, label_dict, corpus, learning_rate, 'resources/training/QBC')
            classifier_two = TextClassifier.load(os.path.join(path_results, 'resources/training/QBC/final-model.pt'))
            pred_mat.append(create_pred_mat_class(unlab_ind, classifier_two, train_text))
        
        else:
            pred_mat = create_pred_mat(unlab_ind, classifier, train_text)

        select_ind = select_next_batch(al_strategy, query_str, label_ind, unlab_ind, args.batch_size, pred_mat)
        label_ind.extend(select_ind)
        unlab_ind = [n for n in unlab_ind if n not in select_ind]

        train_sentences = []
        for idx in label_ind:
            train_sentences.append(datapoints[idx])
        
        train = SentenceDataset(train_sentences)
        corpus = Corpus(train=train)
        label_dict = corpus.make_label_dictionary()
        train_trainer(document_embeddings, label_dict, corpus, args.learning_rate, 'resources/training')
        classifier = TextClassifier.load(os.path.join(path_results, 'resources/training/final-model.pt'))
        
        test_pred = predict_testset(test_text, classifier)

        #choose better performance metric later for class imbalance
        #acc = accuracy(test_labels, test_pred)
        f1_score = f1(test_labels, test_pred, average, minority)
        num_instances.append(len(label_ind))
        performance.append(f1_score)


    run_dict = {"runtime" :timeit.default_timer() - starttime}
    write_json(run_dict, path_results, len(run_dict), "a")
    report = prec_rec_f1(test_labels, test_pred, classes)
    write_json(report, path_results, len(report), "a")
    #conf_mat(test_labels, test_pred, classes, path_results)
    dict_perf = dict(zip(num_instances, performance))
    write_json(dict_perf, path_results, len(dict_perf), "a")

    return num_instances, performance, label_ind
    

def main_func():

    #initialize results
    config = {
        "query_strategy": query_str,
        "word_embedding": word_embedding,
        "neural network": "rnn",
        "dataset": dataset_path_original,
        "learning_rate": args.learning_rate,
        "max epoch": max_epoch,
        "mini batch size": mini_batch_size,
        "stopping_criterion" : stopping_crit,
        "seed" : args.seed,
        "query batch size" : args.batch_size
        }

    write_json(config, path_results, len(config), "w")


    #if query_str not in query_strategy:
        #print("Please choose one of the following strategies:", query_strategy)
    #if word_embedding not in we_embeddings:
        #print("Please choose one of the following word embeddings:", we_embeddings)
    #check if datapaths exists
    #if query_str == "QueryInstanceBMDR" or query_str == "QueryInstanceSPAL":
    #check if cvxpy installed

    #create text, label lists and sentence datapoints
    train_file = os.path.join(dataset_path_original, 'train.tsv')
    train_text, train_labels = create_text_label_list(train_file)
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    #y = [float(i) for i in train_labels]
    #datapoints_all = create_sentence_dataset(train_text, train_labels)

    #load word embeddings, create document embeddings and construct feature matrix
    word_embeddings = [select_word_embedding(word_embedding)]
    document_embeddings = DocumentRNNEmbeddings(word_embeddings,hidden_size=512,reproject_words=True,reproject_words_dimension=256,)
    #feat_mat = create_feat_mat(train_text, document_embeddings)

    #initialize Active Learning
    #alibox = ToolBox(X=feat_mat, y=y, query_type='AllLabels', saving_path='.')

    
    
    cv = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=False)
    kfold_label_idx = []
    overall_perf = {}
    seed_label = 5
    for train_index, test_index in cv.split(train_text, train_labels):
        X_train, X_test = train_text[train_index], train_text[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        datapoints = create_sentence_dataset(X_train, y_train)
        #y = [float(i) for i in y_train]
        y = le.transform(y_train)
        feat_mat = create_feat_mat(X_train, document_embeddings)
        idx = list(range(0, len(X_train)))

        dict_instances = {"train_length": len(X_train), "test_length": len(X_test)}
        write_json(dict_instances, path_results, len(dict_instances), "a")

        #initialize Active Learning
        alibox = ToolBox(X=feat_mat, y=y, query_type='AllLabels', saving_path='.')
       
        label_ind, unlab_ind = select_random(seed_label, idx, 30)
        seed_label = seed_label + 1

        seed_set_labels = list(y_train[label_ind])
        dict_seedset = dict(zip(label_ind, seed_set_labels))
        write_json(dict_seedset, path_results, len(dict_seedset), "a")

        al_strategy = select_query_strategy(alibox, query_str, idx)

        num_instances, performance, label_ind_new = al_main_loop(alibox, al_strategy, document_embeddings, X_train, y_train, X_test, y_test, datapoints, label_ind, unlab_ind)
        kfold_label_idx.append(label_ind_new)
        

        for i in range(len(num_instances)):
            key = str(num_instances[i])
            if key in overall_perf:
                overall_perf[key] = overall_perf[key] + performance[i]
            else:
                overall_perf[key] = performance[i]
           
    print(len(kfold_label_idx))
    #final analysis
    for i in overall_perf:
        overall_perf[i] = overall_perf[i]/5
    df_perform = pd.DataFrame(overall_perf.items(), columns=['num_instances','performance'])
    df_perform.to_csv(os.path.join(path_results, "df_perform.tsv"), sep = '\t')

    
    label_dict = {}
    for i in range(len(kfold_label_idx)):
        name = "kfold" + str(i+1)
        label_dict[name] = list(kfold_label_idx[i])
    df_label_idx = pd.DataFrame(label_dict)
    df_label_idx.to_csv(os.path.join(path_results, "df_label_idx_per_kfold.tsv"), sep = '\t')

main_func()
        

