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


def LearningRateScheduler():
    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        learning_rate = 0.2
        if epoch > 10:
            learning_rate = 0.02
        if epoch > 20:
            learning_rate = 0.01
        if epoch > 50:
            learning_rate = 0.005

        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


