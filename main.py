import json
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from siamese_model import SiameseModel
from utils.preprocessing import *
from utils.logging import *

models = {'siamese': SiameseModel}

def update_model_params(df):
    max_len =  max(df.q1_length.max(), df.q2_length.max())
    model_params['max_len'] = max_len
    model_params['embedding_matrix'] = embedding_matrix
    model_params['num_unique_words'] = embedding_matrix.shape[0]
    model_params['embedding_dim'] = embedding_matrix.shape[1]

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required = True,
                        help='files with the parameters for the experiment')
    args = parser.parse_args()
    config = json.load(open(args.config_file))
    globals().update(config)
    data, labels, embedding_matrix = get_data_and_labels(data_file,nrows = 10000)
    update_model_params(data)
    X_train, X_test, Y_train, Y_test= train_val_test_split(data,labels)
    network = models[model_params['model_type']]
    network = network(model_params)
    network.model.summary()
    print('start training')
    q1_vecs = np.array([np.array(row) for row in X_train['q1_vec'].values])
    q2_vecs = np.array([np.array(row) for row in X_train['q2_vec'].values])
    train_inputs = [q1_vecs ,q2_vecs,X_train['q1_length'].values,X_train['q2_length'].values]
    q1_vecs_test = np.array([np.array(row) for row in X_test['q1_vec'].values])
    q2_vecs_test = np.array([np.array(row) for row in X_test['q2_vec'].values])
    test_inputs = [q1_vecs_test ,q2_vecs_test,X_test['q1_length'].values,X_test['q2_length'].values]

    logdir = get_log_dir()
    tensorboard_callback = TensorboardCallback(logdir)
    early_stopping = EarlyStopping(min_delta=0.001)
    Y_train = np.array([np.array([1, 0]) if l == 0 else np.array([0, 1]) for l in Y_train.values.tolist()])

    history = network.model.fit(train_inputs, Y_train,validation_split=0.2,epochs=100,\
                  batch_size=256,callbacks=[tensorboard_callback,early_stopping], verbose=1)
    print('end training')
    y_pred = network.model.predict(test_inputs)
    Y_test = np.array([np.array([1, 0]) if l == 0 else np.array([0, 1]) for l in Y_test.values.tolist()])

    print('Accuracy score %r' % (network.model.evaluate(test_inputs,Y_test,callbacks=[tensorboard_callback])))
