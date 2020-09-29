from sklearn.datasets import make_classification
#import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy

X, y = make_classification(n_samples=1500, n_features=10, n_informative=5, n_redundant=2,
                           n_repeated=0, n_classes=5, n_clusters_per_class=2, weights=[0.9, 0.5, 0.1, 0.1, 0.1], flip_y=0.01, class_sep=1.0,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

print(type(y))
unique, counts = numpy.unique(y, return_counts=True)
count = dict(zip(unique, counts))
print(count)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0, verbose=4, perplexity = 30,  init="pca", method= 'exact')
tsne_obj= tsne.fit_transform(X)

tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'class':y})
print(tsne_df.head())


sc_plot = sns.scatterplot(x="X", y="Y",
                            hue="class",
                            palette=sns.color_palette("hls", 5),
                            data=tsne_df)

fig = sc_plot.get_figure()
out_dir = '../results/artificial_class/temp'
fig.savefig(os.path.join(out_dir, "scatter_plot_all.png"))



