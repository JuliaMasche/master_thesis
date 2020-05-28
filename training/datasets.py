
def get_dataset_info(name:str):

    if name == "SST-2_50":
        dataset_path_original = "../data/SST-2/tsv/perc_50_50"
        out_dir = '../results/SST-2/perc_50_50'
        classes = ['0', '1']
        sep = "\t"

    elif name == "SST-2_60":
        dataset_path_original = "../data/SST-2/tsv/perc_60_40"
        out_dir = '../results/SST-2/perc_60_40'
        classes = ['0', '1']
        sep = "\t"

    elif name == "SST-2_70":
        dataset_path_original = "../data/SST-2/tsv/perc_70_30"
        out_dir = '../results/SST-2/perc_70_30'
        classes = ['0', '1']
        sep = "\t"
    
    elif name == "SST-2_80":
        dataset_path_original = "../data/SST-2/tsv/perc_80_20"
        out_dir = '../results/SST-2/perc_80_20'
        classes = ['0', '1']
        sep = "\t"

    elif name == "SST-2_90":
        dataset_path_original = "../data/SST-2/tsv/perc_90_10"
        out_dir = '../results/SST-2/perc_90_10'
        classes = ['0', '1']
        sep = "\t"

    elif name == "news":
        dataset_path_original = "../data/news/tsv"
        out_dir = '../results/news'
        classes = ['rec.sport.hockey', 'comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.electronics', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'rec.sport.baseball', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'talk.politics.mideast', 'talk.religion.misc', 'talk.politics.guns', 'comp.windows.x', 'alt.atheism', 'sci.crypt', 'sci.space', 'talk.politics.misc', 'rec.autos', 'misc.forsale']
        sep = "\t"

    elif name == "webkb":
        dataset_path_original = "../data/webkb/tsv"
        out_dir = '../results/webkb'
        classes = ['other', 'project', 'department', 'course', 'staff', 'faculty', 'student']
        sep = "\t"

    elif name == "movie":
        dataset_path_original = "../data/movie/tsv"
        out_dir = '../results/movie'
        classes = ['neg', 'pos']
        sep = "\t"

    #if name == "yelp":
        #dataset_path_original = "../data/yelp/tsv"
        #dataset_path_al = '../data/yelp/al'
        #out_dir = '../results/yelp'
        #classes = ['0', '1']
        #sep = ","
    
    return dataset_path_original, out_dir, classes, sep