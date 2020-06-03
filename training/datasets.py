
def get_dataset_info(name:str):

    if name == "SST-2_original":
        dataset_path_original = "../data/SST-2/tsv/original"
        out_dir = '../results/SST-2/original'
        classes = ['0', '1']
        minority = '0'
        average = 'binary'
        sep = "\t"

    elif name == "SST-2_70":
        dataset_path_original = "../data/SST-2/tsv/500_70_30"
        out_dir = '../results/SST-2/500_70_30'
        classes = ['0', '1']
        minority = '0'
        average = 'binary'
        sep = "\t"
    
    elif name == "SST-2_80":
        dataset_path_original = "../data/SST-2/tsv/500_80_20"
        out_dir = '../results/SST-2/500_80_20'
        classes = ['0', '1']
        minority = '0'
        sep = "\t"

    elif name == "SST-2_90":
        dataset_path_original = "../data/SST-2/tsv/500_90_10"
        out_dir = '../results/SST-2/500_90_10'
        classes = ['0', '1']
        minority = '0'
        average = 'binary'
        sep = "\t"

    elif name == "news_1":
        dataset_path_original = "../data/news/tsv/keep_classes_1"
        out_dir = '../results/news/keep_classes_1'
        classes = ['other', 'comp.windows.x']
        minority = 'comp.windows.x'
        average = 'binary'
        sep = "\t"

    elif name == "news_2":
        dataset_path_original = "../data/news/tsv/keep_classes_2"
        out_dir = '../results/news/keep_classes_2'
        classes = ['other', 'comp.windows.x', 'alt.atheism']
        minority = ['comp.windows.x', 'alt.atheism']
        average = 'weighted'
        sep = "\t"

    elif name == "news_3":
        dataset_path_original = "../data/news/tsv/keep_classes_3"
        out_dir = '../results/news/keep_classes_3'
        classes = ['other', 'comp.windows.x', 'alt.atheism', 'sci.space']
        minority = ['comp.windows.x', 'alt.atheism', 'sci.space']
        average = 'weighted'
        sep = "\t"

    elif name == "news_4":
        dataset_path_original = "../data/news/tsv/keep_classes_4"
        out_dir = '../results/news/keep_classes_4'
        classes = ['other', 'comp.windows.x', 'alt.atheism', 'sci.space', 'talk.politics.mideast']
        minority = ['comp.windows.x', 'alt.atheism', 'sci.space', 'talk.politics.mideast']
        average = 'weighted'
        sep = "\t"

    elif name == "webkb":
        dataset_path_original = "../data/webkb/tsv"
        out_dir = '../results/webkb'
        classes = ['other', 'project', 'course', 'faculty', 'student']
        minority = ['project', 'course', 'faculty', 'student']
        average = 'weighted'
        sep = "\t"

    elif name == "movie_60":
        dataset_path_original = "../data/movie/tsv/500_60_40"
        out_dir = '../results/movie/perc_60_40'
        classes = ['neg', 'pos']
        minority = ["pos"]
        average = 'binary'
        sep = "\t"

    elif name == "movie_80":
        dataset_path_original = "../data/movie/tsv/500_80_20"
        out_dir = '../results/movie/perc_80_20'
        classes = ['neg', 'pos']
        minority = ["pos"]
        average = 'binary'
        sep = "\t"


    return dataset_path_original, out_dir, classes, sep, minority, average