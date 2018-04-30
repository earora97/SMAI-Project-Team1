
# coding: utf-8

# In[1]:

from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from math import ceil
from req import *
from mag_loss import *
from mag_func import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


# In[2]:


class MNISTEncoder(object):
    def __init__(self, emb_dim, sess):
        self.emb_dim = emb_dim
        self.sess = sess
        
        self.inputs = tf.placeholder("float32", [None, 28*28])
        self.labels = tf.placeholder("bool", [None])
        
        self._build_model()

    def _build_model(self):
        # Convolutional encoder
        x_image = tf.reshape(self.inputs, [-1,28,28,1])

        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.W_fc1 = weight_variable([7 * 7 * 64, self.emb_dim])
        self.b_fc1 = bias_variable([self.emb_dim])

        h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.emb = tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1

        # L2 normalize
        self.norm_emb = tf.nn.l2_normalize(self.emb, 1)

    def get_norm_embedding(self, batch):
        return self.sess.run(self.norm_emb, feed_dict={self.inputs: batch})        
        
    def get_embedding(self, batch):
        return self.sess.run(self.emb, feed_dict={self.inputs: batch})


# In[3]:


# Define magnet loss parameters
m = 2
d = 2
k = 5
alpha = 0.5
batch_size = m * d

# Define training data
X = mnist.train.images
y = mnist.train.labels

# Define model and training parameters
emb_dim = 2
n_epochs = 1
epoch_steps = int(ceil(float(X.shape[0]) / batch_size)) 
n_steps = epoch_steps * n_epochs
cluster_refresh_interval = epoch_steps


sess = tf.InteractiveSession()

# Model
with tf.variable_scope('model'):
    model = MNISTEncoder(emb_dim, sess)

# Loss
with tf.variable_scope('magnet_loss'):
    class_inds = tf.placeholder(tf.int32, [m*d])
    train_loss, losses = minibatch_magnet_loss(model.emb, class_inds, m, d, alpha)

train_op = tf.train.AdamOptimizer(1e-4).minimize(train_loss)

sess.run(tf.global_variables_initializer())


# Get initial embedding
extract = lambda x: sess.run(model.emb, feed_dict={model.inputs: x})
initial_reps = compute_reps(extract, X, 400)


# Create batcher
batch_builder = ClusterBatchBuilder(mnist.train.labels, k, m, d)
batch_builder.update_clusters(initial_reps)

batch_losses = []
for i in range(n_steps):
    
    # Sample batch and do forward-backward
    batch_example_inds, batch_class_inds = batch_builder.gen_batch()
    feed_dict = {model.inputs: X[batch_example_inds], class_inds: batch_class_inds}
    _, batch_loss, batch_example_losses =         sess.run([train_op, train_loss, losses], feed_dict=feed_dict)
    
    # Update loss index
    batch_builder.update_losses(batch_example_inds, batch_example_losses)
    
    batch_losses.append(batch_loss)
    if not i % 200:
        print(i, batch_loss)
    
    if not i % cluster_refresh_interval:
        print('Refreshing clusters')
        reps = compute_reps(extract, X, 400)
        batch_builder.update_clusters(reps)
        
final_reps = compute_reps(extract, X, 400)
    
sess.close()
tf.reset_default_graph()

# Plot loss curve
plot_smooth(batch_losses)


# In[4]:


n_plot = 500
imgs = mnist.train.images[:n_plot]
imgs = np.reshape(imgs, [n_plot, 28, 28])
plot_embedding(initial_reps[:n_plot], mnist.train.labels[:n_plot])


# In[5]:


n_plot = 500
imgs = mnist.train.images[:n_plot]
imgs = np.reshape(imgs, [n_plot, 28, 28])
plot_embedding(final_reps[:n_plot], mnist.train.labels[:n_plot])

