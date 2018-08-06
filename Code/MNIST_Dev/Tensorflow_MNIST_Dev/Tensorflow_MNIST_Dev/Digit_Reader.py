###############################################################
'''
This is a summary with information...
'''
###############################################################
import tensorflow as tf

###############################################################
MODEL_PATH = "/tmp/model/model.ckpt"
n_classes = 10

###############################################################
#create a Fully Connected Layer
def FC_layer(input, size_in, size_out, name="FC"):
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([size_in, size_out], stddev=0.1), name='Weights')
        b = tf.Variable(tf.zeros([size_out]), name='Bias')
        ret = tf.matmul(input, w)+b
        #To add summay information about weights, biases and activations:
        #######################################
        '''
        tf.summary.histogram("Weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", ret)
        '''
        #########################################
        return ret

#Set up NN model
def neural_network_model_setup(learning_rate, decay_rate_flag, act_func):
    sess = tf.Session()
    tf.reset_default_graph()

    # Setup placeholders, and reshape the data:
    x = tf.placeholder(tf.float32, [None,784], name="input")
    image = tf.reshape(x[:10], [-1, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10], name="labels")
    if decay_rate_flag == True:
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, 0.95, staircase=True)

    fc1 = eval(act_func)(FC_layer(x, 784, 500, "fc1"))
    fc2 = eval(act_func)(FC_layer(fc1, 500, 250, "fc2"))
    fc3 = eval(act_func)(FC_layer(fc2, 250, 100, "fc3"))
    fc_out = tf.nn.softmax(FC_layer(fc3, 100, n_classes, "fc_out"))

    with tf.name_scope("cost"):
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_out, labels=y), name="cost")
        cost = -tf.reduce_sum(y * tf.log(fc_out))


    with tf.name_scope("train"):
        if decay_rate_flag == False:
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        else:
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
            #train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    with tf.name_scope("confusion_matrix"):
        #cm = tf.contrib.metrics.confusion_matrix(y[0], fc_out[0])
        cm = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(fc_out,1))
        #tf.summary.text(cm)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(fc_out,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

   # sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_PATH)
        
'''
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        iterations = int(mnist.train.num_examples/BATCH_SIZE)
        avg_cost = 0
        for i in range(iterations):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if i % 5 == 0:
                [train_accuracy, c, s] = sess.run([accuracy, cost, summ], feed_dict={x:batch[0], y:batch[1]})
            sess.run(train_step, feed_dict={x:batch[0], y:batch[1]})
            avg_cost += c / iterations
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    with sess.as_default():
        print("Test Accuracy:", (accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))
'''

def main():

    neural_network_model_setup(5E-3, True, 'tf.nn.relu')

###############################################################
if __name__ == '__main__':
    main()

