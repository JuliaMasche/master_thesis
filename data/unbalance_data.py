import preprocess_datasets
import pandas as pd 
import argparse
import os


def binary_class(df, major_class, percentage):
    min_class = ((major_class*100)/percentage) - major_class
    return min_class



def multi_class(df, keep_list):
    for line in range(len(df)):
        if df['label'][line] not in keep_list:
            df['label'][line] = 'other'
    return df



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
parser.add_argument("--path", type = str)
parser.add_argument('-p', "--percentage", default = 60, type = int)
parser.add_argument('-keep','--keep_classes', nargs='+', help='<Required> Set flag')
args = parser.parse_args()


#for i in range(len(paths)):
df = pd.read_csv(os.path.join(args.path, "train.tsv"), sep = '\t')

if args.dataset == "SST-2":
    index_label_1 = df[df['label'] == 1].index.tolist()
    index_label_0 = df[df['label'] == 0].index.tolist()
    df_length = len(df)
    length_1 = len(index_label_1)
    length_0 = len(index_label_0)
    min_class = int(binary_class(df_length, length_1, args.percentage))
    index_label_0_drop = index_label_0[min_class:]
    df = df.drop(index = index_label_0_drop)
    new_path = args.path + '/perc_' + str(args.percentage) + '_' + str(100-args.percentage)
    os.makedirs(new_path, exist_ok = True)
    df.to_csv(os.path.join(new_path, 'train.tsv'), sep ='\t', index = False)
    length, dist, dist_percent, classes = preprocess_datasets.get_info(new_path)

if args.dataset == "movie":
    index_label_neg = df[df['label'] == 'neg'].index.tolist()
    index_label_pos = df[df['label'] == 'pos'].index.tolist()
    df_length = len(df)
    length_neg = len(index_label_neg)
    length_pos = len(index_label_pos)
    min_class = int(binary_class(df_length, length_neg, args.percentage))
    index_label_pos_drop = index_label_pos[min_class:]
    df = df.drop(index = index_label_pos_drop)
    new_path = args.path + '/perc_' + str(args.percentage) + '_' + str(100-args.percentage)
    os.makedirs(new_path, exist_ok = True)
    df.to_csv(os.path.join(new_path, 'train.tsv'), sep ='\t', index = False)
    length, dist, dist_percent, classes = preprocess_datasets.get_info(new_path)


if args.dataset == "news":
    df = multi_class(df, args.keep_classes)
    new_path = args.path + '/keep_classes_' + str(len(args.keep_classes))
    os.makedirs(new_path, exist_ok = True)
    df.to_csv(os.path.join(new_path, 'train.tsv'), sep ='\t', index = False)
    length, dist, dist_percent, classes = preprocess_datasets.get_info(new_path)
