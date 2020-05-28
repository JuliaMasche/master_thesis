from flair.embeddings import WordEmbeddings, FlairEmbeddings, BertEmbeddings, ELMoEmbeddings, FastTextEmbeddings



def select_word_embedding(we_name:str):

    if we_name == "glove":
        word_embeddings = WordEmbeddings('glove')
    
    elif we_name == "flair":
        word_embeddings = FlairEmbeddings('news-forward')

    elif we_name == "bert":
        word_embeddings = BertEmbeddings()

    elif we_name == "fasttext":
        word_embeddings = FastTextEmbeddings('/path/to/local/custom_fasttext_embeddings.bin')
    
    else:
        word_embeddings = ELMoEmbeddings()
    
    return word_embeddings





