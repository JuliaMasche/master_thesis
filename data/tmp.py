import argparse
from preprocess_datasets import distribution_info
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sn
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str)
parser.add_argument("--path", type = str)
args = parser.parse_args()


plt.rcdefaults()
fig, ax = plt.subplots(figsize=(20,10))
#ax.figure(figsize=(20,10))
df = pd.read_csv(os.path.join(args.path, "train.tsv"), sep = '\t')
l = len(df)
dist = distribution_info(df).to_dict()
for x in dist:
    dist[x] = round((dist[x]/l) * 100, 2)
y_pos = np.arange(len(dist.keys()))
ax.barh(y_pos, dist.values(), align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(dist.keys())
#pd.DataFrame({'label':dist.index, 'num':dist.values})
values = dist.values()
#ax = dist.plot.barh(x= dist.index, y= dist.values)
ax.set_xlim(0, 100)
ax.set_ylabel("classes")
ax.set_xlabel("dist in percentage")
for i, v in enumerate(values):
    ax.text(v + 0.5, i - 0.15, str(v) + '%', color='blue', fontweight='bold')
plt.savefig(os.path.join(args.path,'class_dist_percentage.png'))
plt.show()
