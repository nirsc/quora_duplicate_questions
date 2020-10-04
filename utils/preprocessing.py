import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re
from utils.embeddings import *
from sklearn.model_selection import train_test_split

#read and organize data

def read_data(data_file,nrows = None, delim = '\t'):
    # df = pd.read_csv(data_file, index_col = 'id', nrows=nrows, delimiter = delim)
    df = pd.read_hdf(data_file,'quora_duplicate_questions',mode='a')
    if nrows is not None:
        return df.sample(nrows)
    return df

def clean_text_data(df):
    question_1 = []
    question_2 = []
    for row in df.iterrows():
        row = row[1]
        question_1.append(clean_sentence(row.question1))
        question_2.append(clean_sentence(row.question2))
    df['question1'] = question_1
    df['question2'] = question_2
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



def sent_to_vec_df(df, method = 'concat'):
    question_1 = []
    question_2 = []
    max_question_len = max(df.q1_length.max(), df.q2_length.max())
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
    embedding_matrix = create_embedding_matrix(word_index)
    return df, embedding_matrix




def clean_sentence(sent):
    if not isinstance(sent, str):
        return [""]
    # remove non-alphabetic characters
    sent = re.sub("[^a-zA-Z]", " ", sent)

    # tokenize the sentences
    sent = word_tokenize(sent.lower())
    sent = [w for word in sent for w in word.split('_')]
    # lemmatize each word to its lemma
    sent = [lemmatizer.lemmatize(i) for i in sent]

    return sent


def get_data_and_labels(data_path,nrows = None):
    data = read_data(data_path, nrows)
    data = clean_text_data(data)
    data = add_sent_lengths(data)
    data, embedding_matrix = sent_to_vec_df(data)

    return data[['q1_vec','q2_vec','q1_length','q2_length']],data['is_duplicate'], embedding_matrix


def train_val_test_split(data,labels,test_size = 0.333):
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size= test_size, stratify=labels,\
                                                        random_state=42)

    return X_train, X_test, Y_train, Y_test

DATA_DIR = '..//data//'
filename = 'quora_duplicate_questions.tsv'
# df = read_data(DATA_DIR+filename,nrows = 2)
# df = clean_text_data(df)
# df = add_sent_lengths(df)
# vec_txt_df = sent_to_vec_df(df)
# # for row in df.iterrows():
# #     row = row[1]
# #     print(row.question1)
# # # # #     print(row.question2)


