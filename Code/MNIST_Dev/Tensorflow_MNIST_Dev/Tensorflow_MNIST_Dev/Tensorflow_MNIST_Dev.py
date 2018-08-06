import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
n_classes = 10
BATCH_SIZE = 100
training_epochs = 20
display_epoch = 1
###############################################################
CUSTOM_NODES = True
LOGDIR = "/tmp/mnist_tutorial/final/"
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

######################################################################################################################
#using set node counts for neural network model building:
######################################################################################################################
######################################################################################################################
def neural_network_model(learning_rate, use_fc_num, hparam, act_func, node_num):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data:
    x = tf.placeholder(tf.float32, [None,784], name="input")
    image = tf.reshape(x[:10], [-1, 28, 28, 1])
    tf.summary.image("image", image)
    y = tf.placeholder(tf.float32, [None, 10], name="labels")
    #global_step = tf.Variable(0)
    #learning_rate = tf.train.exponential_decay(learning_rate, global_step, 50, 0.75, staircase=True)
    
    if use_fc_num == 5:
        fc1 = eval(act_func)(FC_layer(x, 784, node_num, "fc1"))
        fc2 = eval(act_func)(FC_layer(fc1, node_num, node_num, "fc2"))
        fc3 = eval(act_func)(FC_layer(fc2, node_num, node_num, "fc3"))
        fc4 = eval(act_func)(FC_layer(fc3, node_num, node_num, "fc4"))
        fc5 = eval(act_func)(FC_layer(fc4, node_num, node_num, "fc5"))
        fc_out = tf.nn.softmax(FC_layer(fc5, node_num, n_classes, "fc_out"))

    if use_fc_num == 3:
        fc1 = eval(act_func)(FC_layer(x, 784, node_num, "fc1"))
        fc2 = eval(act_func)(FC_layer(fc1, node_num, node_num, "fc2"))
        fc3 = eval(act_func)(FC_layer(fc2, node_num, node_num, "fc3"))
        fc_out = tf.nn.softmax(FC_layer(fc3, node_num, n_classes, "fc_out"))

    if use_fc_num == 2:
        fc1 = eval(act_func)(FC_layer(x, 784, node_num, "fc1"))
        fc2 = eval(act_func)(FC_layer(fc1, node_num, node_num, "fc2"))
        fc_out = tf.nn.softmax(FC_layer(fc2, node_num, n_classes, "fc_out"))

    if use_fc_num == 1:
        fc1 = eval(act_func)(FC_layer(x, 784, node_num, "fc1"))
        fc_out = tf.nn.softmax(FC_layer(fc1, node_num, n_classes, "fc_out"))

    with tf.name_scope("cost"):
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_out, labels=y), name="cost")
        cost = -tf.reduce_sum(y * tf.log(fc_out))
        tf.summary.scalar("cost", cost)

    with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

    with tf.name_scope("confusion_matrix"):
        #cm = tf.contrib.metrics.confusion_matrix(y[0], fc_out[0])
        cm = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(fc_out,1))
        #tf.summary.text(cm)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(fc_out,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    for i in range(4000):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x:batch[0], y:batch[1]})
            writer.add_summary(s,i)
        sess.run(train_step, feed_dict={x:batch[0], y:batch[1]})
    with sess.as_default():
        print("Test Accuracy:", (accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))

def make_hparam_string(learning_rate, use_fc_num, act_func, node_num):
    fc_param = "fc=" + str(use_fc_num)
    node_num_param = "nn_count="+str(node_num)
    act_func_param = "act_func=" + str(act_func)
    return "lr_%.0E, %s, %s, %s" % (learning_rate, fc_param, node_num_param, act_func_param)
########################################################################################################################
#custom node count neural network model:
########################################################################################################################
def neural_network_model_custom(learning_rate, use_fc_num, hparam, decay_rate_flag, act_func):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data:
    x = tf.placeholder(tf.float32, [None,784], name="input")
    image = tf.reshape(x[:10], [-1, 28, 28, 1])
    tf.summary.image("image", image)
    y = tf.placeholder(tf.float32, [None, 10], name="labels")
    if decay_rate_flag == True:
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, 0.95, staircase=True)
    
    if use_fc_num == 5:
        fc1 = eval(act_func)(FC_layer(x, 784, 500, "fc1"))
        fc2 = eval(act_func)(FC_layer(fc1, 500, 500, "fc2"))
        fc3 = eval(act_func)(FC_layer(fc2, 500, 250, "fc3"))
        fc4 = eval(act_func)(FC_layer(fc3, 250, 100, "fc4"))
        fc5 = eval(act_func)(FC_layer(fc4, 100, 60, "fc5"))
        fc_out = tf.nn.softmax(FC_layer(fc5, 60, n_classes, "fc_out"))

    if use_fc_num == 3:
        fc1 = eval(act_func)(FC_layer(x, 784, 500, "fc1"))
        fc2 = eval(act_func)(FC_layer(fc1, 500, 250, "fc2"))
        fc3 = eval(act_func)(FC_layer(fc2, 250, 100, "fc3"))
        fc_out = tf.nn.softmax(FC_layer(fc3, 100, n_classes, "fc_out"))


    if use_fc_num == 2:
        fc1 = eval(act_func)(FC_layer(x, 784, 500, "fc1"))
        fc2 = eval(act_func)(FC_layer(fc1, 500, 250, "fc2"))
        fc_out = tf.nn.softmax(FC_layer(fc2, 250, n_classes, "fc_out"))

    if use_fc_num == 1:
        fc1 = eval(act_func)(FC_layer(x, 784, 500, "fc1"))
        fc_out = tf.nn.softmax(FC_layer(fc1, 500, n_classes, "fc_out"))

    with tf.name_scope("cost"):
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_out, labels=y), name="cost")
        cost = -tf.reduce_sum(y * tf.log(fc_out))
        tf.summary.scalar("cost", cost)

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
        tf.summary.scalar("accuracy", accuracy)
    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        iterations = int(mnist.train.num_examples/BATCH_SIZE)
        avg_cost = 0
        for i in range(iterations):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if i % 5 == 0:
                [train_accuracy, c, s] = sess.run([accuracy, cost, summ], feed_dict={x:batch[0], y:batch[1]})
                writer.add_summary(s,epoch*iterations+i)
            sess.run(train_step, feed_dict={x:batch[0], y:batch[1]})
            avg_cost += c / iterations
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    with sess.as_default():
        print("Test Accuracy:", (accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))

def make_hparam_string_custom(learning_rate, use_fc_num, decay_rate_flag, act_func):
    fc_param = "fc=" + str(use_fc_num)
    drf_param = "drf=" + str(decay_rate_flag)
    act_func_param = "act_func=" + str(act_func)
    return "lr_%.0E, %s, %s, %s" % (learning_rate, fc_param,drf_param,act_func_param)
######################################################################################################################
######################################################################################################################

def main():
    #activation functions to call, otherwise default sigmoid:
    activation = {'relu': 'tf.nn.relu',
          'tanh': 'tf.nn.tanh',
          'sigmoid': 'tf.nn.sigmoid',
          'elu': 'tf.nn.elu',
          'softplus': 'tf.nn.softplus',
          'softsign': 'tf.nn.softsign',
          'relu6': 'tf.nn.relu6',
          'dropout': 'tf.nn.dropout'}
    for activation_function in ['relu']:
        if activation_function in activation:
            act_func = activation[eval('activation_function')]
        else:
            act_func = 'other'
        #number of nodes in the layers:
        if CUSTOM_NODES == False:
            for node_num in [250, 500]:
                #learning rate in scientific notation:
                for learning_rate in [5E-3]:
                    #number of fully connected layers:
                    for use_fc_num in [1,2,3,5]:
                        #decaying rate not used for this testing:
                        decay_rate_flag = False
                        hparam = make_hparam_string(learning_rate, use_fc_num, activation_function, node_num)
                        print('Starting run for %s' % hparam)
                        neural_network_model(learning_rate, use_fc_num, hparam, act_func, node_num)
        else: 
        #learning rate in scientific notation:
            for learning_rate in [5E-3]:
                #number of fully connected layers:
                for use_fc_num in [3]:
                    #learning rate starting values to include in decaying rate:
                    if learning_rate in [5E-3]:
                        decay_rate_flag = True
                    else:
                        decay_rate_flag = False

                    hparam = make_hparam_string_custom(learning_rate, use_fc_num, decay_rate_flag, activation_function)
                    print('Starting run for %s' % hparam)
                    neural_network_model_custom(learning_rate, use_fc_num, hparam, decay_rate_flag, act_func)

    print('Done Training!')
    print('Run  ">>tensorboard --logdir=%s"  to see the results.' % LOGDIR)
            

if __name__ == '__main__':
    main()
