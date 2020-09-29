from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import string

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings
from flair.data import Corpus, Sentence

from datasets import get_dataset_info
from word_embeddings import select_word_embedding


sets = ["SST-2_original", "SST-2_90", "SST-2_80", "SST-2_70", "SST-2_60", "SST-2_50", "wiki_1000", "wiki_2000", "webkb_1000", "webkb_2000", "news_1000", "news_2000", "news_1500", "movie_70", "movie_80"]
we_embeddings = ['glove', 'flair', 'fasttext', 'bert', 'word2vec', 'elmo_small', 'elmo_medium', 'elmo_original']

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=sets, default = "SST-2_70", type = str)
parser.add_argument("-we", "--word_embedding", choices=we_embeddings, default = "glove", type = str)
args = parser.parse_args()
#dataset_path_original, out_dir, classes, sep, minority, average = get_dataset_info(args.dataset)
path = '/home/julia/projects/master_thesis/data/wiki/tsv/dataset_1000_without_unknown/train.tsv'

def create_feat_mat(text, document_embeddings):
    matrix = []
    for i in range(len(text)):
        print(i)
        sentence = Sentence(text[i])
        document_embeddings.embed(sentence)
        embedding = sentence.get_embedding().cpu().detach().numpy()
        matrix.append(embedding)
    feat_mat = np.asarray(matrix)
    return feat_mat

#train_file = os.path.join(dataset_path_original, 'train.tsv') 
#df = pd.read_csv(train_file, sep = '\t')
df = pd.read_csv(path, sep = '\t')


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
#table = str.maketrans('', '', string.punctuation)
#X= [w.translate(table) for w in X]
labels = df['label'].tolist()
labels = [str(i) for i in labels]
X = np.asanyarray(X)
classes = set(labels)
le = preprocessing.LabelEncoder()
le.fit(list(classes))
y = le.transform(labels)

word_embeddings = select_word_embedding(args.word_embedding)
document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased', fine_tune=True)

feat_mat = create_feat_mat(X, document_embeddings)

#feat_mat_reduced = PCA(n_components=30, random_state = 0).fit_transform(feat_mat)
#feat_mat_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(feat_mat)
# Create the visualizer and draw the vectors
tsne = TSNE(n_components=2, random_state=0, verbose=4, perplexity = 30,  init="pca", method= 'exact')
tsne_obj= tsne.fit_transform(feat_mat)

tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'class':y})
print(tsne_df.head())
"""
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs= tsne_df['X'], 
    ys= tsne_df['Y'], 
    zs= tsne_df['Z'], 
    c= tsne_df['class'], 
    cmap='tab10'
)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
"""



sc_plot = sns.scatterplot(x="X", y="Y",
                            hue="class",
                            palette=sns.color_palette("hls", len(classes)),
                            data=tsne_df)

fig = sc_plot.get_figure()
out_dir = '/home/julia/projects/master_thesis/data/wiki/tsv/dataset_1000_without_unknown'
#os.makedirs(out_dir, exist_ok = True)
fig.savefig(os.path.join(out_dir, "scatter_plot_all.png"))
