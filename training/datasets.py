
def get_dataset_info(name:str):

    if name == "SST-2_original":
        dataset_path_original = "../data/SST-2/tsv/original/filtered"
        out_dir = '../results/SST-2/original'
        classes = ['0', '1']
        minority = '0'
        average = 'binary'
        sep = "\t"

    elif name == "SST-2_50":
        dataset_path_original = "../data/SST-2/tsv/3000_50_50"
        out_dir = '../results/SST-2/3000_50_50'
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
        dataset_path_original = "../data/SST-2/tsv/3000_80_20"
        out_dir = '../results/SST-2/3000_80_20'
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

    elif name == "news_original":
        dataset_path_original = "../data/news/tsv/original/"
        out_dir = '../results/news/original/'
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

    elif name == "news_3000":
        dataset_path_original = "../data/news/tsv/dataset_3000/"
        out_dir = '../results/news/dataset_3000/'
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

    elif name == "news_balanced":
        dataset_path_original = "../data/news/tsv/dataset_1000/balanced"
        out_dir = '../results/news/dataset_1000/balanced'
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

    elif name == "wiki_original":
        dataset_path_original = "../data/wiki/tsv/original"
        out_dir = '../results/wiki/original'
        classes = ["romance", "horror", "western",  "unknown", "film noir", "crime", "crime drama", "musical", "drama", "thriller", "action","mystery", "comedy","adventure"]
        minority = ["romance", "horror", "western", "film noir", "crime", "crime drama", "musical", "thriller", "action","mystery", "comedy","adventure"]
        average = 'weighted'
        sep = "\t"

    elif name == "wiki_1000":
        dataset_path_original = "../data/wiki/tsv/dataset_1000"
        out_dir = '../results/wiki/dataset_1000'
        classes = ["romance", "horror", "western",  "unknown", "film noir", "crime", "crime drama", "musical", "drama", "thriller", "action","mystery", "comedy","adventure"]
        minority = ["romance", "horror", "western", "film noir", "crime", "crime drama", "musical", "thriller", "action","mystery", "comedy","adventure"]
        average = 'weighted'
        sep = "\t"

    elif name == "wiki_3000":
        dataset_path_original = "../data/wiki/tsv/dataset_3000"
        out_dir = '../results/wiki/dataset_3000'
        classes = ["romance", "horror", "western",  "unknown", "film noir", "crime", "crime drama", "musical", "drama", "thriller", "action","mystery", "comedy","adventure"]
        minority = ["romance", "horror", "western", "film noir", "crime", "crime drama", "musical", "thriller", "action","mystery", "comedy","adventure"]
        average = 'weighted'
        sep = "\t"

    elif name == "wiki_1000_wo_unknown":
        dataset_path_original = "../data/wiki/tsv/dataset_1000_without_unknown"
        out_dir = '../results/wiki/dataset_1000_without_unknown'
        classes = ["drama",
        "comedy",
        "western",
        "crime",
        "action",
        "thriller",
        "romance",
        "horror",
        "adventure",
        "musical",
        "mystery",
        "film noir",
        "crime drama",
        "comedy, drama",
        "romantic comedy",
        "musical comedy",
        "comedy drama",
        "serial",
        "sci-fi",
        "war",
        "comedy-drama",
        "family",
        "documentary",
        "fantasy"]
        minority= ["adventure",
        "musical",
        "mystery",
        "film noir",
        "crime drama",
        "comedy, drama",
        "romantic comedy",
        "musical comedy",
        "comedy drama",
        "serial",
        "sci-fi",
        "war",
        "comedy-drama",
        "family",
        "documentary",
        "fantasy"]        
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


    elif name == "scan":
        dataset_path_original = "../data/Scan/tsv/original_short_sequences"
        out_dir = '../results/Scan/original_short_sequences'
        classes = [
        "kv_vitalleistung",
        "urem_rechnung",
        "fed_ruecktritt_kuendigung",
        "uv_ruecktritt_kuendigung",
        "oeamtc_kv_antrag",
        "rv_sach_antrag",
        "kv_akut_versorgt_leistung",
        "lv_drittrechte",
        "kv_kuendigung",
        "flv_faelligkeit",
        "ld_servicecenter_eingehende_korrespondenz",
        "kv_leistung",
        "ah_regulierung",
        "rv_abbucher",
        "uits_rechnung",
        "rechnung",
        "uv_schadenmeldung",
        "fed_vertrag",
        "kfz_schaden",
        "provisionsnote",
        "kv_arztbericht",
        "kfz_bonus_malus"]
        minority = [
        "uv_ruecktritt_kuendigung",
        "kv_kuendigung",
        "ah_regulierung",
        "rv_sach_antrag",
        "uv_schadenmeldung",
        "fed_ruecktritt_kuendigung"
        ]
        average = 'weignted'
        sep = "\t"


    return dataset_path_original, out_dir, classes, sep, minority, average