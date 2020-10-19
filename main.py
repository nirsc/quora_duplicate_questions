import json
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from siamese_model import SiameseModel
from utils.preprocessing import *
from utils.logging import *
import pickle


def update_model_params():
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
    model_params = config.get('model_params')
    max_len = model_params.get('max_len',None)
    data_file = config['data_file']
    checkpoint_dir = config.get('checkpoint_dir','./checkpoint/')
    X_train = pd.read_hdf('data/X_train.h5', 'data')
    X_val = pd.read_hdf('data/X_val.h5', 'data')
    X_test = pd.read_hdf('data/X_test.h5', 'data')
    Y_train = pd.read_hdf('data/Y_train.h5', 'data')
    Y_val = pd.read_hdf('data/Y_val.h5', 'data')
    Y_test = pd.read_hdf('data/Y_test.h5', 'data')
    embedding_matrix = pickle.load(open('data/embedding_matrix.pkl', 'rb'))
    update_model_params()
    network = SiameseModel(model_params)
    load_weights = config.get('load_weights',None)
    ckpt = config.get('checkpoint',None)
    initial_epoch = 0
    
    if load_weights:
        if not ckpt:
            ckpt = tf.train.latest_checkpoint('./checkpoint')
        initial_epoch = int(ckpt.split('.')[1][-1])
        network.model.load_weights(ckpt)

    train_inputs,Y_train = prepare_data_for_training(X_train,Y_train)
    val_inputs, Y_val = prepare_data_for_training(X_val, Y_val)
    test_inputs, Y_test = prepare_data_for_training(X_test, Y_test)
    logdir = get_log_dir()
    tensorboard_callback = TensorboardCallback(logdir)
    model_checkpoint = ModelCheckPoint(checkpoint_dir)
    print("start training")
    history = network.model.fit(train_inputs, Y_train,initial_epoch=initial_epoch, validation_data=(val_inputs,Y_val),epochs=initial_epoch+10,\
                  batch_size=256,callbacks=[tensorboard_callback,model_checkpoint], verbose=1)
    print('end training')
    y_pred = network.model.predict(test_inputs)
    print('Accuracy score %r' % (network.model.evaluate(test_inputs,Y_test,callbacks=[tensorboard_callback])))
