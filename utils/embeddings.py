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
        if nlp(word).has_vector:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = nlp(word).vector
    return embedding_matrix


def get_glove_vecs(sentence_list):
    sent_list_length = len(sentence_list)
    glove_wv_mat= np.zeros((sent_list_length,300))
    for i in tqdm(range(sent_list_length)):
        sent = sentence_list[i]
        tokens = nlp(sent)
        vector_sum = np.zeros(300)
        for token in tokens:
            if token.has_vector:
                vector_sum+=token.vector
        glove_wv_mat[i,:]=vector_sum
    return glove_wv_mat


def concat_embed(sent,maxlen):
    embedding = np.empty(0)
    tokens = nlp(sent)
    for token in tokens:
        if token.has_vector:
            embedding = np.concatenate([embedding, token.vector])
        else:
            embedding = np.concatenate(embedding, np.zeros(300))
    temp_embed = np.zeros(maxlen*300)
    temp_embed[:len(embedding)] = embedding
    embedding = temp_embed
    return embedding

def sum_embed(sent):
    embedding = np.zeros(300)
    tokens = nlp(sent)
    for token in tokens:
        if token.has_vector:
            embedding += token.vector
    return embedding

def mean_embed(sent):
    embedding = sum_embed(sent)
    sent_length = len(sent)
    embedding/=sent_length
    return embedding


"""
returns the glove embedding of the sentence. 
arguments:
1)'method' - determines how to join the vectors 
for every token. options are:
'concat' - concatenate all the vectors.
'sum' - sum of all vectors
'mean' - mean of all vectors.
2)maxlen - relevant only for concatention. Needed 
to make sure all embeddings are same length.  

"""
def get_sentence_embedding(sent,method = 'concat', maxlen =None):

    if method == 'concat':
        return concat_embed(sent,maxlen)
    elif method == 'sum':
        return sum_embed(sent)
    else:
        return mean_embed(sent)





