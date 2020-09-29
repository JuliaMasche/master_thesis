from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.metrics import geometric_mean_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


def prec_rec_f1(y_true, y_pred, classes):
    return classification_report(y_true, y_pred, labels=classes, output_dict=True)


def conf_mat(y_true, y_pred, classes, out_dir):
    mat = confusion_matrix(y_true, y_pred, normalize='true')
    df_cm = pd.DataFrame(mat, index = [i for i in classes],
                  columns = [i for i in classes])
    plt.figure(figsize = (10,7))
    heat = sn.heatmap(df_cm, annot=True)
    fig = heat.get_figure()
    fig.savefig(os.path.join(out_dir, "conf_mat.png"))


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def f1(y_true, y_pred, average, minority):
    if average == 'binary':
        return f1_score(y_true, y_pred, average=average, pos_label=minority)
    else:
        return f1_score(y_true, y_pred, average=average, labels=minority)

def performance_measure(y_true, y_pred, average, measure:str, minority, classes):
    if measure == "f1":
        return f1(y_true, y_pred, average, minority)
    elif measure == "accuracy":
        return accuracy(y_true, y_pred)
    elif measure == "g-mean":
        return geometric_mean_score(y_true, y_pred, average='weighted', labels = classes)


def plot_num_instances_performance(instances, performance, out_dir, label, name):
    for i in range(len(instances)):
        plt.plot(instances[i], performance[i], label = label[i])
    plt.axhline(y= 0.6480061777634462, color='r', linestyle='--', label="no active learning")
    plt.xlabel("Number of instances")
    plt.ylabel("Performance: G-Mean")
    plt.xlim(left =instances[0][0], right = instances[0][-1])
    plt.ylim(bottom=0.4, top = 0.7)
    #plt.xticks(ticks=instances[0])
    plt.title(name)
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'num_instance_performance.png'))

parser = argparse.ArgumentParser()
parser.add_argument('-p','--paths', nargs='+', help='<Required> Set flag')
parser.add_argument('-o','--out', type = str, help='<Required> Set flag')
parser.add_argument('-st','--strategy', nargs='+', help='<Required> Set flag')
parser.add_argument('-n','--name', help='<Required> Set flag', type = str)
args = parser.parse_args()
num_instances = []
performance = []
for p in args.paths:
    df = pd.read_csv(os.path.join(p, 'df_perform.tsv'), sep = '\t')
    num_instances.append(df['num_instances'].tolist())
    performance.append(df['performance'].tolist())
plot_num_instances_performance(num_instances, performance, args.out, args.strategy, args.name)
