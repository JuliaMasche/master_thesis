import random
from sklearn.cluster import KMeans


def select_random(seed:int, idx_list, size:int):
    random.seed(seed)
    label_idx = sorted(random.sample(idx_list, size))
    unlabel_idx = [n for n in idx_list if n not in label_idx]
    return label_idx, unlabel_idx



def select_clustering(seed, train_text, size, pred_mat, idx_list):
    kmeans = KMeans(n_clusters=size, random_state=seed, init='k-means++').fit(train_text)
    label_idx = kmeans.cluster_centers_ 
    unlabel_idx = [n for n in idx_list if n not in label_idx]
    return label_idx, unlabel_idx