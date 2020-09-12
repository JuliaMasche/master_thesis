from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.metrics import geometric_mean_score
import pandas as pd
import os
import numpy as np
import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import altair as alt


def get_classes_dist(idx, df_original):
    docs = df_original.loc[idx]
    dist = docs['label'].value_counts()
    dist = dist.to_dict()
    print(dist)
    return dist

def get_classes(df):
    classes = set(df['label'].to_list())
    return list(classes)


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    plt.show()
    return axe

def prep_df(df, name):
    df = df.stack().reset_index()
    df.columns = ['c1', 'c2', 'values']
    df['DF'] = name
    return df



parser = argparse.ArgumentParser()
parser.add_argument('-op','--original_path', type= str , help='<Required> Set flag')
parser.add_argument('-p','--paths', nargs='+', help='<Required> Set flag')
parser.add_argument('-o','--out', type = str, help='<Required> Set flag')
args = parser.parse_args()

query = ["Random", "Uncertainty", "QBC", "GraphDensity", "EER", "LAL_iter", "LAL_indep"]

df_original = pd.read_csv(os.path.join(args.original_path, 'train.tsv'), sep = '\t')
classes = get_classes(df_original)
index = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']


df_analysis = pd.DataFrame(columns = ['Strategy', 'Query', 'Class', 'Value'])

for p in range(len(args.paths)):
    #df_analysis = pd.DataFrame(columns = classes, index = ['seed_set', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'])
    i = 0
    name = query[p]
    df = pd.read_csv(os.path.join(args.paths[p], 'df_label_idx_per_kfold.tsv'), sep = '\t')
    kfold1 = df['kfold1'].tolist()
    for x in index:
        idx = kfold1[i: i+100]
        dist = get_classes_dist(idx, df_original)
        for cl in classes:
            if cl in dist:
                df_analysis = df_analysis.append({'Strategy': name, 'Query': x, 'Class': str(cl), 'Value': dist[cl]}, ignore_index=True)
            else:
                df_analysis = df_analysis.append({'Strategy': name, 'Query': x, 'Class': str(cl), 'Value': 0}, ignore_index=True)
        i = i+100
    

    df_analysis.to_csv(os.path.join(args.out, "kfold_analysis_whole.tsv"), sep = '\t')