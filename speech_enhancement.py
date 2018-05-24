import tensorflow as tf
import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import time

###############################################################################
############################ Ideas & notes ####################################
###############################################################################

# - Add spectrum as feature?
# - Add 50% overlap of speech frames? Usually done in speech processing,
#       Pro: more training data
#       Con: RNN should pick up time depency on its own
# - Think about loss, is MSE okay?
#       Pro: Getting as near as possible to ideal files
#       Con: I think there is another option here

# To do:
    # Get steps and reshaping of data right

###############################################################################
############################## Variables ######################################
###############################################################################

# Debugging
DEBUG = 0
LOAD_MODEL = 0

# Network
n_epochs = 31
n_steps = 960 # 48kHz * 20ms
n_neurons = 150
n_input = 1
n_output = 1
n_layers = [512, 256, 128, 64]        # n_neurons for each layer
learning_rate = 0.01
keep_prob = 0.75
n_test_files = 5    # Number of test files, rest is training

# Audio processing variables
window_length = 20e-3   # 20ms window, usual in speech processing
overlap = 50    # percent


###############################################################################
################################# Data ########################################
###############################################################################

# Load data
X_filelist = []
y_filelist = []
filelist_numerator = 0  # since filelists are of the same length only one instance is needed

# Prepare filelists for training and test
for root, dirs, files in os.walk("./X_data/"):
    for name in files:
        X_filelist.append(os.path.join(root, name))
        X_test_filelist = X_filelist[:n_test_files]
        X_filelist = X_filelist[n_test_files:]
for root, dirs, files in os.walk("./y_data/"):
    for name in files:
        y_filelist.append(os.path.join(root, name))
        y_test_filelist = y_filelist[:n_test_files]
        y_filelist = y_filelist[n_test_files:]
if len(y_filelist) != len(X_filelist):
    sys.exit("Error! Mismatch of training data and labels!")

def get_train_data(epoch):
    # if file is finished, reset framecounter and get next one from list
    if file_finished == 1:
        framecounter = 0
        filelist_numerator += 1
        if filelist_numerator >= len(X_filelist)-1: # if filelist is finished
            filelist_numerator = 0  # reset counter
            epoch += 1  # one epoch done
        file_finished = 0
        X_fs, X_data = wavfile.read(filelist[-1])
        X_data = X_data/2147483647 # normalization to [-1, +1]
        y_fs, y_data = wavfile.read(filelist[-1])
        y_data = y_data/2147483647 # normalization to [-1, +1]
        n_samples = X_fs * window_length

    # while file isn't finished, get next frame
    # 20ms = 960 samples for Fs = 48e3
    elif file_finished == 0:
        X = X_data[framecounter*n_samples:(framecounter+window_length)*n_samples]
        y = y_data[framecounter*n_samples:(framecounter+window_length)*n_samples]
        framecounter += 1
        if 2*(framecounter+window_length)*n_samples >= len(data):
            # this leaves out 20ms of the end, but there is silence anyways
            file_finished = 1

    return X, y, epoch


###############################################################################
################################ Misc #########################################
###############################################################################

# Folders for Tensorboard graphs
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)


###############################################################################
################################ Graph ########################################
###############################################################################

# Construct Graph
with tf.name_scope("Input"):
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_input], name="X")
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_output], name="y")
    keep_holder = tf.placeholder(dtype=tf.float32, name="Keep_prob")


cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in n_layers]
cells_dropout = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_holder)
for cell in cells]
stacked_cell = tf.contrib.rnn.MultiRNNCell(cells_dropout)
with tf.name_scope("GRU"):
    rnn_outputs, _ = tf.nn.dynamic_rnn(stacked_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_layers[-1] ])
stacked_outputs = tf.contrib.layers.fully_connected(stacked_rnn_outputs,
n_output, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_output])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

saver = tf.train.Saver()
file_writer = tf.summary.FileWriter("./graphs/", tf.get_default_graph())
loss_summary = tf.summary.scalar("Loss", loss)


###############################################################################
############################### Training ######################################
###############################################################################
epoch = 0
with tf.Session() as sess:
    if LOAD_MODEL==1:
        saver.restore(sess, "./model/mymodel.ckpt")
    else:
        init.run()
    while epoch <= n_epochs:
        for iteration in range(data.shape[0] - n_steps - 1):
            X_data, y_data, epoch = get_train_data(epoch)
            X_data = X_data.reshape((-1, n_steps, n_input))
            y_data = y_data.reshape((-1, n_output))
            sess.run(train_op, feed_dict={X: X_data, y: y_data, keep_holder: keep_prob})

            if iteration%250==0:
                 summary = loss_summary.eval(feed_dict={X: X_data, y: y_data, keep_holder: keep_prob})
                 step = iteration + epoch*data.shape[0]
                 file_writer.add_summary(summary, step)

            if DEBUG==1:
                 # It's messy, I know, but it's quick
                 mse, outs, ys = sess.run([loss, outputs, y], feed_dict={X: X_data, y: y_data, keep_holder: 1.0})
                 print("Output:", outs)
                 print("Y:", ys)
                 print("Output_shape:", outs.shape, "y_shape:", ys.shape)
                 print("Epoch:", epoch, "Iteration:", iteration, ", MSE:", mse, flush=True)
                 input()


        # Each epoch calculate & print training and test error
        mse, outs, ys = sess.run([loss, outputs, y], feed_dict={X: X_data, y: y_data, keep_holder: 1.0})
        print("Output_shape:", outs.shape, "y_shape:", ys.shape)
        print("Output:", outs[-1, -1], "Y:", ys[-1])
        print("Epoch:", epoch, ", MSE:", mse, flush=True)

        # Save model each 10 epochs
        if epoch%10==0:
            saver.save(sess, "./model/mymodel.ckpt")
    # Save model after finishing everything
    saver.save(sess, "./model/mymodel.ckpt")

file_writer.close()
