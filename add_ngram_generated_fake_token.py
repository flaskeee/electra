import tensorflow as tf
if tf.__version__ == '1.15.0':
    tf.enable_eager_execution()


def worker(original_path, out_path):
    with tf.io.TFRecordWriter(out_path) as writer:
        for e in tf.data.TFRecordDataset(original_path):

