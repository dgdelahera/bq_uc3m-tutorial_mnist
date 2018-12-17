import tensorflow as tf
from model import cnn_model_fn
import numpy as np
from PIL import Image

with tf.Graph().as_default():
    image = Image.open("test/3.png").convert('LA')
    image = image.resize([28, 28])
    image = np.invert(image).astype("float32")

    logits = cnn_model_fn(image, 'eval')

    softmax_logits = tf.nn.softmax(logits)
    predictions = tf.nn.top_k(softmax_logits, k=3)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess, 'output/')

        class_number = sess.run(predictions)

        for i in range(0, 3):
            print("Es un", class_number.indices[0][i], "con un ", round(class_number.values[0][i]*100, 2), " %")


