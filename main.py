import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import cnn_model_fn

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 1000
batch_size = 100


def main(argv):

    with tf.Graph().as_default() as g:
        #Placeholders
        training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
        labels = tf.placeholder(tf.float32, [None, labels_size])

        # # Parte 1
        # W = tf.Variable(tf.truncated_normal([image_size*image_size, labels_size], stddev=0.1))
        # b = tf.Variable(tf.constant(0.1, shape=[labels_size]))
        #
        # # Build the network (only output layer)
        # logits = tf.matmul(training_data, W) + b

        logits = cnn_model_fn(training_data)

        # Define the loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # Training step
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Accuracy calculation
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Tensorboard
        accuracy_summary = tf.summary.scalar(name='accuracy', tensor=accuracy)
        loss_summary = tf.summary.scalar(name='loss', tensor=loss)
        train_writer = tf.summary.FileWriter('log', g)
        summaries = tf.summary.merge_all()

        # Saving the model
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(steps_number):
                input_batch, labels_batch = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={training_data: input_batch, labels: labels_batch})

                if i % 10 == 0:
                    train_accuracy, summary = sess.run([accuracy, summaries],
                                                       feed_dict={training_data: input_batch, labels: labels_batch})
                    print("Step %d, training batch accuracy %g %%" % (i, train_accuracy*100))
                    # Tensorboard
                    train_writer.add_summary(summary, i)

            # Evaluate on the test set
            test_accuracy = sess.run(accuracy, feed_dict={training_data: mnist.test.images, labels: mnist.test.labels})
            print("Test accuracy: %g %%" % (test_accuracy*100))

            # Saving the model
            saver.save(sess=sess, save_path='output/')


if __name__ == "__main__":
    tf.app.run()
