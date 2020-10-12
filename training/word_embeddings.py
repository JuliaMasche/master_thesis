from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, ELMoEmbeddings, TransformerDocumentEmbeddings, FastTextEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings



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
        word_embeddings = TransformerWordEmbeddings('bert-base-uncased', layers='-1', fine_tune = True)
        
    elif we_name == "fasttext":
        word_embeddings = WordEmbeddings('en')
    
    return word_embeddings



def select_document_embeddding(name:str, word_embeddings):

    if name == "Pool":
        document_embedding = DocumentPoolEmbeddings([word_embeddings])

    elif name == "RNN":
        document_embedding = DocumentRNNEmbeddings([word_embeddings], hidden_size=256)

    elif name == "Transformer_ger":
        document_embedding = TransformerDocumentEmbeddings('bert-base-german-dbmdz-uncased', fine_tune=True)

    elif name == "Transformer_eng":
        document_embedding = TransformerDocumentEmbeddings('bert-base-uncased', fine_tune=True)

    return document_embedding






