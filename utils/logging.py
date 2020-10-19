import tensorflow as tf
from datetime import datetime


def get_log_dir():
    return "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

def create_file_writer(logdir):
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    return file_writer

def TensorboardCallback(logdir):
    return tf.keras.callbacks.TensorBoard(log_dir=logdir)

def EarlyStopping(min_delta = 0.001):
    return tf.keras.callbacks.EarlyStopping(min_delta = min_delta, mode='max', monitor='val_loss', patience=2)


def ModelCheckPoint(checkpoint_dir  = './checkpoint/', save_freq = 'epoch'):
    checkpoint_path = checkpoint_dir +"/cp-{epoch:04d}.ckpt"
    return tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=False)


