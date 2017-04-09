import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mb_size = 32
X_dim = mnist.train.images.shape[1] # number of features
print "X_dim: ", X_dim
y_dim = mnist.train.labels.shape[1] # number of labels
print "y_dim: ", y_dim
z_dim = 10 # num of classes
h_dim = 128 # hidden layer size
eps = 1e-8

d_lr = 1e-3 # discriminator learning rate
g_lr = 1e-3 # generator learning rate

d_steps = 3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def full_plot(samples_list, itr):
    n = len(samples_list)
    m = len(samples_list[0])

    _, ax = plt.subplots(n, m, sharex=True, sharey=True)
    for i in range(n):
        for j in range(m):
            ax[i][j].imshow(samples_list[i][j].reshape(28,28), 'gray')
            ax[i][j].set_axis_off()
    plt.savefig('out/sample_{}'.format(str(itr).zfill(3)) , dpi=600)
    plt.close()
    return

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def generator(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2_gan = tf.Variable(xavier_init([h_dim, 1]))
D_b2_gan = tf.Variable(tf.zeros(shape=[1]))
D_W2_aux = tf.Variable(xavier_init([h_dim, y_dim]))
D_b2_aux = tf.Variable(tf.zeros(shape=[y_dim]))


def discriminator(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    out_gan = tf.nn.sigmoid(tf.matmul(D_h1, D_W2_gan) + D_b2_gan)
    out_aux = tf.matmul(D_h1, D_W2_aux) + D_b2_aux
    return out_gan, out_aux


theta_G = [G_W1, G_W2, G_b1, G_b2]
theta_D = [D_W1, D_W2_gan, D_W2_aux, D_b1, D_b2_gan, D_b2_aux]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def cross_entropy(logit, y):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


G_sample = generator(z, y)

D_real, C_real = discriminator(X)
D_fake, C_fake = discriminator(G_sample)

# Cross entropy aux loss
C_loss = cross_entropy(C_real, y) + cross_entropy(C_fake, y)

# GAN D loss
D_loss = tf.reduce_mean(tf.log(D_real + eps) + tf.log(1. - D_fake + eps))
DC_loss = -(D_loss + C_loss)

# GAN's G loss
G_loss = tf.reduce_mean(tf.log(D_fake + eps))
GC_loss = -(G_loss + C_loss)

D_solver = (tf.train.AdamOptimizer(learning_rate=d_lr)
            .minimize(DC_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=g_lr)
            .minimize(GC_loss, var_list=theta_G))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

itr = 0

for it in range(1000000):
    X_mb, y_mb = mnist.train.next_batch(mb_size)
    z_mb = sample_z(mb_size, z_dim)

    _, DC_loss_curr = sess.run(
        [D_solver, DC_loss],
        feed_dict={X: X_mb, y: y_mb, z: z_mb}
    )

    _, GC_loss_curr = sess.run(
        [G_solver, GC_loss],
        feed_dict={X: X_mb, y: y_mb, z: z_mb}
    )

    if it % 1000 == 0:

        n_imgs = 10
        
        # Pick all numbers to generate samples from
        c_list = [np.zeros([n_imgs, y_dim]) for _ in range(10)]
        for i in range(10):
            c_list[i][range(n_imgs), i] = 1

        samples_list = [sess.run(G_sample, feed_dict={z: sample_z(n_imgs, z_dim), y: c_list[i]}) for i in range(10)]

        # print('Iter: {}; DC_loss: {:.4}; GC_loss: {:.4}'
        #       .format(it, DC_loss_curr, GC_loss_curr))
        # full_plot(samples_list, itr)

        itr += 1

# Get data
X_train = mnist.train.images
y_train = [np.argmax(one_hot) for one_hot in list(mnist.train.labels)]
X_test = mnist.test.images
y_test = [np.argmax(one_hot) for one_hot in list(mnist.test.labels)]

# Train
if os.path.exists('svm_basic.sav'):
    svm_basic = pickle.load(open('svm_basic.sav', 'rb'))
else:
    svm_basic = SVC()
    print ("Training svm ... ")
    svm_basic.fit(X_train[:1000], y_train[:1000])
    pickle.dump(svm_basic, open("svm_basic.sav", 'wb'))
    print ("Trained and saved!")

# Predict
z_train = svm_basic.predict(X_train[:1000])
# z_test = svm_basic.predict(X_test)
print ("Training Accuracy: {0:.0f}%").format(accuracy_score(y_train[:1000], z_train)*100)
# print ("Test Accuracy: {0:.0f}%").format(accuracy_score(y_test, z_test))

# Append generated data to original train data

