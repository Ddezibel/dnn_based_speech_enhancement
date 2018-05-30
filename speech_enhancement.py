import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from scipy.io import wavfile
import time
from datetime import datetime

###############################################################################
############################ Ideas & notes ####################################
###############################################################################

# - Add spectrum as feature?
# - Add 50% overlap of speech frames? Usually done in speech processing,
#       Pro: more training data
#       Con: RNN should pick up time depency on its own
# - Think about loss, is MSE okay?
#       Pro: Getting as near as possible to ideal files
#       Con: Definitely not ideal, since silent files would produce a very small
#            MSE due to normalization!

# To do:
    # - Implement testing and generation of test files!
    #       -> will do that after the net trained for one weekend
    # - Align y_filelist and X_filelist: While there can be multiple types of noise,
    #       only one underlying label is needed for all of them.
    #       Thus len(y_filelist) = N*len(X_filelist), N>=0. Fix this!
    #       Also mind the test files lists!

# Notes:
    # - Time for one step, no batching:
    #        fetching training data: approx. 0ms
    #       training step of 3 layer model [512, 256, 128]: approx. 1.7s
    #       training of 4 layer model [512, 256, 128, 64]: approx. 3.2s
    #       (trained on Nvidia Geforce GTX 960m)
    # - I don't think batching is worth that much here, neither is batch
    #       normalization, since all files are normalized to start with [-1;+1]

###############################################################################
############################## Variables ######################################
###############################################################################

# Debugging
DEBUG = 1
LOAD_MODEL = 0
TEST = 0

# Network
n_epochs = 10
n_steps = 960 # 48kHz * 20ms
n_input = 1
n_output = 1
n_layers = [960, 960, 960]        # n_neurons for each layer
learning_rate = 0.01
keep_prob = 0.75
n_test_files = 2    # Number of test files, rest is training

# Audio processing variables
window_length = 20e-3   # 20ms window, usual in speech processing
overlap = 50    # percent


###############################################################################
################################# Data ########################################
###############################################################################

# Load data
X_filelist = []
y_filelist = []
X_test_filelist = []
y_test_filelist = []
filelist_numerator = 0  # since filelists are of the same length only one instance is needed
test_filelist_numerator = 0
file_finished = 1
test_file_finished = 1

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
    print("Length of y_filelist:", len(y_filelist), ", Length of X_filelist:", len(X_filelist))
    sys.exit("Error! Mismatch of training data and labels!")

def get_train_data(epoch):
    global file_finished, filelist_numerator, X_filelist, y_filelist, n_samples
    global X_data, y_data, framecounter
    # if file is finished, reset framecounter and get next one from list
    if file_finished == 1:
        framecounter = 0
        filelist_numerator += 1
        if filelist_numerator >= len(X_filelist)-1: # if filelist is finished
            filelist_numerator = 0  # reset counter
            epoch += 1  # one epoch done
        file_finished = 0
        X_fs, X_data = wavfile.read(X_filelist[filelist_numerator])
        X_data = X_data/2147483647 # normalization to [-1, +1]
        y_fs, y_data = wavfile.read(y_filelist[filelist_numerator])
        y_data = y_data/2147483647 # normalization to [-1, +1]
        n_samples = X_fs * window_length

    # while file isn't finished, get next frame
    # 20ms = 960 samples for Fs = 48e3
    if file_finished == 0:
        X = X_data[int(framecounter*n_samples):int((framecounter+1)*n_samples)]
        y = y_data[int(framecounter*n_samples):int((framecounter+1)*n_samples)]
        framecounter += 1
        if (framecounter+2)*n_samples >= len(X_data):
            # this leaves out 20ms of the end, but there is silence anyways
            file_finished = 1

    return X, y, epoch, filelist_numerator

# This can be written way better, by combining it with get_train_data,
# and passing a variable to draw the data from the list of test files instead
# of training files. Also namescopes would be really helpful here
def get_test_data():
    global test_file_finished, test_filelist_numerator, X_test_filelist, y_test_filelist, n_samples
    global X_test_data, y_test_data, framecounter, test_finished, X_test_fs
    # if file is finished, reset framecounter and get next one from list
    if test_file_finished == 1:
        framecounter = 0
        test_filelist_numerator += 1
        if test_filelist_numerator > len(X_test_filelist)-1: # if filelist is finished
            test_filelist_numerator = 0  # reset counter
            test_finished = 1
        test_file_finished = 0
        X_test_fs, X_test_data = wavfile.read(X_test_filelist[test_filelist_numerator])
        X_test_data = X_test_data/2147483647 # normalization to [-1, +1]
        y_fs, y_test_data = wavfile.read(y_test_filelist[test_filelist_numerator])
        y_test_data = y_test_data/2147483647 # normalization to [-1, +1]
        n_samples = X_test_fs * window_length

    # while file isn't finished, get next frame
    # 20ms = 960 samples for Fs = 48e3
    if test_file_finished == 0:
        X_test = X_test_data[int(framecounter*n_samples):int((framecounter+1)*n_samples)]
        y_test = y_test_data[int(framecounter*n_samples):int((framecounter+1)*n_samples)]
        framecounter += 1
        if (framecounter+2)*n_samples >= len(X_test_data):
            # this leaves out 20ms of the end, but there is silence anyways
            test_file_finished = 1

    return X_test, y_test, epoch, test_filelist_numerator, test_finished, X_test_fs


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
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_output], name="y")
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
step = 1
old_filelist_numerator = 1
old_test_filelist_numerator = 1
old_epoch = 0
data_counter = 1
new_file = 0
new_test_file = 0
with tf.Session() as sess:
    if LOAD_MODEL==1:
        saver.restore(sess, "./model/mymodel.ckpt")
    else:
        init.run()

    while epoch <= n_epochs:
        X_feed_data, y_feed_data, epoch, filelist_numerator = get_train_data(epoch)
        X_feed_data = X_feed_data.reshape((-1, n_steps, n_input))
        y_feed_data = y_feed_data.reshape((-1, n_steps, n_output))
        sess.run(train_op, feed_dict={X: X_feed_data, y: y_feed_data, keep_holder: keep_prob})

        # Check for new file being processed
        if old_filelist_numerator != filelist_numerator:
            new_file = 1    # Flag, if new file is being processed
        old_filelist_numerator = filelist_numerator

        # Check for new epoch
        if old_epoch != epoch:
            new_epoch = 1
        old_epoch = epoch

        if new_file==1:
             summary = loss_summary.eval(feed_dict={X: X_feed_data, y: y_feed_data, keep_holder: keep_prob})
             step += 1
             file_writer.add_summary(summary, step)
             new_file = 0

        if DEBUG==1:
             # It's messy, I know, but it's quick
             mse, outs, ys = sess.run([loss, outputs, y], feed_dict={X: X_feed_data, y: y_feed_data, keep_holder: 1.0})
             #print("Outs:", outs)
             #print("y:", ys)
             #print("Output_shape:", outs.shape, "y_shape:", ys.shape)
             print("Epoch:", epoch, ", File:", filelist_numerator, ", MSE:", mse, flush=True)
             #input()

        new_epoch = 1

############################### Test ######################################
        # Each epoch run a test and generate data
        if new_epoch==1:
            test_finished = 0
            wavefile = 0
            while test_finished == 0:
                X_test_data, y_test_data, _, test_filelist_numerator, test_finished, X_test_fs = get_test_data()
                if test_finished == 0:
                    X_test_data = X_feed_data.reshape((-1, n_steps, n_input))
                    y_test_data = y_feed_data.reshape((-1, n_steps, n_output))
                    # calculate test error
                    mse, outs, ys = sess.run([loss, outputs, y], feed_dict={X: X_test_data, y: y_test_data, keep_holder: 1.0})
                    # but also save generated wav files
                    wavefile = np.append(wavefile, outs)

                if old_test_filelist_numerator != test_filelist_numerator:
                    new_test_file = 1    # Flag, if new file is being processed
                old_test_filelist_numerator = test_filelist_numerator

                # If a file is finished being generated, write it
                if new_test_file==1:
                    wavefile = wavefile * 2147483647 # rescale
                    filename = './generated/epoch_%d_data_%d.wav' % (epoch, data_counter)
                    wavfile.write(filename, X_test_fs, wavefile)    # save
                    data_counter+=1
                    new_test_file = 0
                    wavefile = 0
            data_counter = 1


        # Save model each N processed files
        if (filelist_numerator%5==0) and (new_file==1):
            saver.save(sess, "./model/mymodel.ckpt")
            print("Model saved.", flush=True)
            new_file = 0
    # Save model after finishing everything
    saver.save(sess, "./model/mymodel.ckpt")

file_writer.close()
