import pandas as pd
#from training.datasets import get_dataset_info
import json
import os
import copy
import argparse


datasets = ["SST-2","webkb", "movie", "news"]
paths = ["/home/julia/projects/master_thesis_al/data/SST-2/tsv/",
        "/home/julia/projects/master_thesis_al/data/webkb/tsv/",
        "/home/julia/projects/master_thesis_al/data/movie/tsv/",
        "/home/julia/projects/master_thesis_al/data/news/tsv/"]


def distribution_info(df):
    dist = df['label'].value_counts()
    dist = dist.to_dict()
    dist_percent = copy.deepcopy(dist)
    length = len(df)
    for x in dist_percent:
        dist_percent[x] = (dist_percent[x]/length) * 100
    return dist, dist_percent

def get_classes(df):
    classes = set(df['label'].to_list())
    return classes

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def write_json(dictionary, path, size, arg):
    # Serializing json  
    json_object = json.dumps(dictionary, indent = size, default=set_default) 
  
    # Writing to sample.json 
    with open(os.path.join(path, "info.json"), arg) as outfile: 
        outfile.write(json_object)


def get_info(path):
    df = pd.read_csv(os.path.join(path, "train.tsv"), sep = '\t')
    d = {}
    length = len(df)
    dist, dist_percent = distribution_info(df)
    classes = get_classes(df)
    d.update({'num_instances': length})
    d.update({'distribution': dist})
    d.update({'distribution_percentage' : dist_percent})
    d.update({'classes': classes})
    write_json(d, path, len(d), 'w')
    return length, dist, dist_percent, classes


parser = argparse.ArgumentParser()
parser.add_argument("--path", type = str)
args = parser.parse_args()
df = pd.read_csv(os.path.join(args.path, "train.tsv"), sep = '\t')
length, dist, dist_percent, classes = get_info(args.path)
cut = 500

for cl in classes:
    index_label = df[df['label'] == cl].index.tolist()
    new_index_len = int((dist_percent[cl]*cut)/100)
    index_label_drop = index_label[new_index_len:]
    df = df.drop(index = index_label_drop)

new_path = args.path + '/short_train_set'
os.makedirs(new_path, exist_ok = True)
df.to_csv(os.path.join(new_path, 'train.tsv'), sep ='\t', index = False)
length, dist, dist_percent, classes = get_info(new_path)

