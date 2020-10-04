from __future__ import absolute_import
from __future__ import division
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow.keras.backend as k
    from tensorflow import keras
    from tensorflow.keras import Model,Input
    from tensorflow.keras.layers import LSTM, Multiply, subtract, concatenate, Dense, Embedding
    from tensorflow.keras.backend import sum as tf_sum

class SiameseModel(object):
    """
    Implements a Siamese LSTM network.
    """

    def __init__(self,model_params):

        super(SiameseModel, self).__init__()
        max_len = model_params['max_len']
        embedding_matrix = model_params['embedding_matrix']
        num_unique_words = model_params['num_unique_words']
        embedding_dim = model_params['embedding_dim']
        self.sentence_1 = Input(shape=(None,))
        self.sentence_2 = Input(shape=(None,))
        self.sentence_1_embedding = Embedding(num_unique_words,embedding_dim,\
                                              embeddings_initializer=keras.initializers.Constant(embedding_matrix),\
                                              trainable=False)(self.sentence_1)
        self.sentence_2_embedding = Embedding(num_unique_words,embedding_dim,\
                                              embeddings_initializer=keras.initializers.Constant(embedding_matrix),\
                                              trainable=False)(self.sentence_2)
        self.len_sent_1 = Input(shape=(None, 1), dtype='int32')
        self.len_sent_2 = Input(shape=(None, 1), dtype='int32')
        #self.squared_diff = Multiply([tf_sum(self.difference),tf_sum(self.difference)])

        self.shared_lstm = LSTM(1000)
        self.question_1_encoding = self.shared_lstm(self.sentence_1_embedding)
        self.question_2_encoding = self.shared_lstm(self.sentence_2_embedding)
        self.Hadamard = Multiply()([self.question_1_encoding, self.question_2_encoding])
        self.difference = subtract([self.question_1_encoding, self.question_2_encoding])
        self.squared_euclidean_distance = tf_sum(Multiply()([self.difference, self.difference]),axis=1)
        self.squared_euclidean_distance = k.expand_dims(self.squared_euclidean_distance, axis=-1)
        self.features = concatenate([self.question_1_encoding, self.question_2_encoding,\
                                  self.Hadamard, self.squared_euclidean_distance],)
        self.h1 = Dense(1100,activation='relu')(self.features)
        self.h2 = Dense(800, activation='relu')(self.h1)
        self.out = Dense(2, activation='softmax')(self.h2)
        self.model = Model(inputs = [self.sentence_1,self.sentence_2,self.len_sent_1,self.len_sent_2], outputs=[self.out])
        self.model.compile(loss="binary_crossentropy", optimizer= 'adam', metrics=['binary_accuracy'])
