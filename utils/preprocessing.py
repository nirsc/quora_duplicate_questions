import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re
from utils.embeddings import *
from time import time
from sklearn.model_selection import train_test_split
#read and organize data

def read_data(data_file,nrows = None, delim = '\t'):
    df = pd.read_hdf(data_file,'quora_duplicate_questions',mode='a')
    df.dropna(subset = ['question1','question2','is_duplicate'])
    if nrows is not None:
        return df.sample(nrows)
    return df

def clean_text_data(df):
    start = time()
    question_1 = []
    question_2 = []

    for row in df.iterrows():
        row = row[1]
        question_1.append(clean_sentence(row.question1))
        question_2.append(clean_sentence(row.question2))
    df['question1'] = question_1
    df['question2'] = question_2
    print('cleaning text data took {} seconds'.format(time()-start))
    return df

def add_sent_lengths(df):
    q1_lengths = []
    q2_lengths = []
    for row in df.iterrows():
        q1_length = len(row[1].question1)
        q2_length = len(row[1].question2)
        q1_lengths.append(q1_length)
        q2_lengths.append(q2_length)
    df['q1_length'] = q1_lengths
    df['q2_length'] = q2_lengths
    return df


def sent_to_vec_df(df, max_len=None):
    question_1 = []
    question_2 = []
    max_question_len = max_len if max_len else max(df.q1_length.max(), df.q2_length.max())
    for row in df.iterrows():
        q1 = row[1].question1
        q1 = " ".join(q1)
        q2 = row[1].question2
        q2 = " ".join(q2)
        question_1.append(q1)
        question_2.append(q2)
    tokenizer = tokenize(question_1 + question_2)
    word_index = tokenizer.word_index
    question_1 = pad_sequences(tokenizer.texts_to_sequences(question_1),max_question_len).tolist()
    question_2 = pad_sequences(tokenizer.texts_to_sequences(question_2),max_question_len).tolist()
    df['q1_vec'] = question_1
    df['q2_vec'] = question_2
    start = time()
    embedding_matrix = create_embedding_matrix(word_index)
    print('creating embedding matrix took {} seconds'.format(time()-start))
    return df, embedding_matrix



def clean_sentence(sent):
    if not isinstance(sent, str):
        return [""]
    # remove non-alphabetic characters
    sent = re.sub("[^a-zA-Z]", " ", sent)

    # tokenize the sentences
    sent = word_tokenize(sent.lower())
    sent = [w for word in sent for w in word.split('_')]
    stop_words = set(stopwords.words('english'))
    sent = [w for w in sent if w not in stop_words]
    # lemmatize each word to its lemma
    sent = [lemmatizer.lemmatize(i) for i in sent]

    return sent

def get_data_and_labels(data_path,nrows = None,max_len = None):
    data = read_data(data_path, nrows)
    data = clean_text_data(data)
    data = add_sent_lengths(data)
    data, embedding_matrix = sent_to_vec_df(data,max_len)

    return data[['q1_vec','q2_vec','q1_length','q2_length']],data['is_duplicate'], embedding_matrix


def train_val_test_split(data,labels,test_size = 0.333,val_size = 0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size= test_size, stratify=labels,\
                                                        random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, stratify=Y_train, \
                                                        random_state=42)

    X_train.to_hdf('data/X_train.h5','data')
    X_val.to_hdf('data/X_val.h5','data')
    X_test.to_hdf('data/X_test.h5','data')
    Y_train.to_hdf('data/Y_train.h5','data')
    Y_val.to_hdf('data/Y_val.h5','data')
    Y_test.to_hdf('data/Y_test.h5','data')
    return X_train, X_val, X_test, Y_train, Y_val ,Y_test


def prepare_data_for_training(data,labels):
    q1_vecs = np.array([np.array(row) for row in data['q1_vec'].values])
    q2_vecs = np.array([np.array(row) for row in data['q1_vec'].values])
    inputs = [q1_vecs ,q2_vecs,data['q1_length'].values,data['q2_length'].values]
    labels = np.array([np.array([1, 0]) if l == 0 else np.array([0, 1]) for l in labels.values.tolist()])
    return inputs,labels

