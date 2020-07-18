
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

    elif name == "news_1000":
        dataset_path_original = "../data/news/tsv/dataset_1000/"
        out_dir = '../results/news/dataset_1000/'
        classes = [
        "alt.atheism",
        "sci.electronics",
        "sci.crypt",
        "comp.sys.mac.hardware",
        "rec.sport.baseball",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
        "sci.space",
        "comp.os.ms-windows.misc",
        "comp.graphics",
        "talk.religion.misc",
        "rec.sport.hockey",
        "misc.forsale",
        "comp.windows.x",
        "rec.motorcycles",
        "talk.politics.misc",
        "soc.religion.christian",
        "sci.med",
        "talk.politics.guns",
        "rec.autos"
    ]
        minority = [
        "alt.atheism",
        "sci.electronics",
        "sci.crypt",
        "comp.sys.mac.hardware",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
        "sci.space",
        "comp.os.ms-windows.misc",
        "talk.religion.misc",
        "rec.sport.hockey",
        "comp.windows.x",
        "talk.politics.misc",
        "soc.religion.christian",
        "sci.med",
        "talk.politics.guns",
        "rec.autos"
    ]
        average = 'weighted'
        sep = "\t"

    elif name == "news_2000":
        dataset_path_original = "../data/news/tsv/dataset_2000/"
        out_dir = '../results/news/dataset_2000/'
        classes = [
        "alt.atheism",
        "sci.electronics",
        "sci.crypt",
        "comp.sys.mac.hardware",
        "rec.sport.baseball",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
        "sci.space",
        "comp.os.ms-windows.misc",
        "comp.graphics",
        "talk.religion.misc",
        "rec.sport.hockey",
        "misc.forsale",
        "comp.windows.x",
        "rec.motorcycles",
        "talk.politics.misc",
        "soc.religion.christian",
        "sci.med",
        "talk.politics.guns",
        "rec.autos"
    ]
        minority = [
        "alt.atheism",
        "sci.electronics",
        "sci.crypt",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
        "sci.space",
        "comp.os.ms-windows.misc",
        "talk.religion.misc",
        "rec.sport.hockey",
        "comp.windows.x",
        "talk.politics.misc",
        "soc.religion.christian",
        "sci.med",
        "talk.politics.guns",
        "rec.autos"
    ]
        average = 'weighted'
        sep = "\t"

    elif name == "news_1500":
        dataset_path_original = "../data/news/tsv/dataset_1500/"
        out_dir = '../results/news/dataset_1500/'
        classes = [
        "alt.atheism",
        "sci.electronics",
        "sci.crypt",
        "comp.sys.mac.hardware",
        "rec.sport.baseball",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
        "sci.space",
        "comp.os.ms-windows.misc",
        "comp.graphics",
        "talk.religion.misc",
        "rec.sport.hockey",
        "misc.forsale",
        "comp.windows.x",
        "rec.motorcycles",
        "talk.politics.misc",
        "soc.religion.christian",
        "sci.med",
        "talk.politics.guns",
        "rec.autos"
    ]
        minority = [
        "alt.atheism",
        "sci.electronics",
        "sci.crypt",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
        "sci.space",
        "comp.os.ms-windows.misc",
        "talk.religion.misc",
        "rec.sport.hockey",
        "comp.windows.x",
        "talk.politics.misc",
        "soc.religion.christian",
        "sci.med",
        "talk.politics.guns",
        "rec.autos"
    ]
        average = 'weighted'
        sep = "\t"


    elif name == "webkb_1000":
        dataset_path_original = "../data/webkb/tsv/dataset_1000"
        out_dir = '../results/webkb/dataset_1000'
        classes = ['other', 'project', 'course', 'faculty', 'student', 'staff', 'department']
        minority = ['project', 'course', 'faculty', 'student', 'staff', 'department']
        average = 'weighted'
        sep = "\t"

    elif name == "wiki_1000":
        dataset_path_original = "../data/wiki/tsv/dataset_1000"
        out_dir = '../results/wiki/dataset_1000'
        classes = ["romance", "horror", "western",  "unknown", "film noir", "crime", "crime drama", "musical", "drama", "thriller", "action","mystery", "comedy","adventure"]
        minority = ["romance", "horror", "western", "film noir", "crime", "crime drama", "musical", "thriller", "action","mystery", "comedy","adventure"]
        average = 'weighted'
        sep = "\t"

    elif name == "wiki_2000":
        dataset_path_original = "../data/wiki/tsv/dataset_2000"
        out_dir = '../results/wiki/dataset_2000'
        classes = ["romance", "horror", "western",  "unknown", "film noir", "crime", "crime drama", "musical", "drama", "thriller", "action","mystery", "comedy","adventure"]
        minority = ["romance", "horror", "western", "film noir", "crime", "crime drama", "musical", "thriller", "action","mystery", "comedy","adventure"]
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