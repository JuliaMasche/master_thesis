
def get_dataset_info(name:str):

    if name == "SST-2_original":
        dataset_path_original = "../data/SST-2/tsv/original"
        out_dir = '../results/SST-2/original'
        classes = ['0', '1']
        minority = '0'
        average = 'binary'
        sep = "\t"

    elif name == "SST-2_50":
        dataset_path_original = "../data/SST-2/tsv/5000_50_50"
        out_dir = '../results/SST-2/5000_50_50'
        classes = ['0', '1']
        minority = '0'
        average = 'binary'
        sep = "\t"

    elif name == "SST-2_70":
        dataset_path_original = "../data/SST-2/tsv/1000_70_30"
        out_dir = '../results/SST-2/1000_70_30'
        classes = ['0', '1']
        minority = '0'
        average = 'binary'
        sep = "\t"
    
    elif name == "SST-2_80":
        dataset_path_original = "../data/SST-2/tsv/1000_80_20"
        out_dir = '../results/SST-2/1000_80_20'
        classes = ['0', '1']
        minority = '0'
        sep = "\t"

    elif name == "SST-2_90":
        dataset_path_original = "../data/SST-2/tsv/1000_90_10"
        out_dir = '../results/SST-2/1000_90_10'
        classes = ['0', '1']
        minority = '0'
        average = 'binary'
        sep = "\t"

    elif name == "news_3":
        dataset_path_original = "../data/news/tsv/keep_classes_3"
        out_dir = '../results/news/keep_classes_3'
        classes = ['other', 'comp.windows.x', 'alt.atheism', 'sci.space']
        minority = ['comp.windows.x', 'alt.atheism', 'sci.space']
        average = 'weighted'
        sep = "\t"

    elif name == "news_1000_4":
        dataset_path_original = "../data/news/tsv/dataset_1000/keep_classes_4"
        out_dir = '../results/news/dataset_1000/keep_classes_4'
        classes = ["rec.autos", "other", "comp.os.ms-windows.misc", "sci.space", "talk.politics.guns"]
        minority = ["rec.autos", "comp.os.ms-windows.misc", "sci.space", "talk.politics.guns"]
        average = 'weighted'
        sep = "\t"

    elif name == "news_2000_4":
        dataset_path_original = "../data/news/tsv/dataset_2000/keep_classes_4"
        out_dir = '../results/news/dataset_2000/keep_classes_4'
        classes = ["rec.autos", "other", "comp.os.ms-windows.misc", "sci.space", "talk.politics.guns"]
        minority = ["rec.autos", "comp.os.ms-windows.misc", "sci.space", "talk.politics.guns"]
        average = 'weighted'
        sep = "\t"

    elif name == "webkb_1000":
        dataset_path_original = "../data/webkb/tsv/dataset_1000"
        out_dir = '../results/webkb/dataset_1000'
        classes = ['other', 'project', 'course', 'faculty', 'student', 'staff', 'department']
        minority = ['project', 'course', 'faculty', 'student', 'staff', 'department']
        average = 'weighted'
        sep = "\t"

    elif name == "webkb_2000":
        dataset_path_original = "../data/webkb/tsv/dataset_2000"
        out_dir = '../results/webkb/dataset_2000'
        classes = ['other', 'project', 'course', 'faculty', 'student', 'staff', 'department']
        minority = ['project', 'course', 'faculty', 'student', 'staff', 'department']
        average = 'weighted'
        sep = "\t"

    elif name == "movie_70":
        dataset_path_original = "../data/movie/tsv/1000_70_30"
        out_dir = '../results/movie/1000_70_30'
        classes = ['neg', 'pos']
        minority = ["pos"]
        average = 'binary'
        sep = "\t"

    elif name == "movie_70":
        dataset_path_original = "../data/movie/tsv/1000_80_20"
        out_dir = '../results/movie/1000_80_20'
        classes = ['neg', 'pos']
        minority = "pos"
        average = 'binary'
        sep = "\t"


    return dataset_path_original, out_dir, classes, sep, minority, average