from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, ELMoEmbeddings, FastTextEmbeddings, StackedEmbeddings



def select_word_embedding(we_name:str):

    if we_name == "glove":
        word_embeddings = WordEmbeddings('glove')
    
    elif we_name == "flair":
        word_embeddings = StackedEmbeddings([
                                        WordEmbeddings('glove'),
                                        FlairEmbeddings('news-forward'),
                                        FlairEmbeddings('news-backward'),
                                       ])
    elif we_name == "bert":
        word_embeddings = TransformerWordEmbeddings('bert-base-uncased')
        
    elif we_name == "fasttext":
        word_embeddings = WordEmbeddings('en')
    
    elif we_name == "elmo_small":
        word_embeddings = ELMoEmbeddings('small')

    elif we_name == "elmo_medium":
        word_embeddings = ELMoEmbeddings('medium')

    elif we_name == "elmo_original":
        word_embeddings = ELMoEmbeddings('orignal')
    
    return word_embeddings





