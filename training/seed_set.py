import random



def select_random(seed:int, idx_list, size:int):
    random.seed(seed)
    label_idx = sorted(random.sample(idx_list, size))
    unlabel_idx = [n for n in idx_list if n not in label_idx]
    return label_idx, unlabel_idx
