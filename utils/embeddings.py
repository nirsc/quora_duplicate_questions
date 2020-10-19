from tqdm import  tqdm
import spacy.cli
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from tensorflow.keras.preprocessing.sequence import pad_sequences

nlp = spacy.load('en_core_web_lg')

def tokenize(texts, MAX_WORDS = None):
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer




def create_embedding_matrix(word_index,embedding_dim = 300):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        token = nlp(word,disable = ['tagger','parser','ner','textcat'])
        if token.has_vector:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = token.vector
        else:
            embedding_matrix[i] = np.random.normal(0,0.01,300)
    return embedding_matrix






