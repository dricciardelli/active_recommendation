import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from sklearn.metrics import accuracy_score


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

mb_size = 32
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
z_dim = 10
h_dim = 128
eps = 1e-8

d_lr = 1e-3
g_lr = 1e-3

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
    plt.savefig('multi-out/sample_{}'.format(str(itr).zfill(3)) , dpi=600)
    plt.close()
    return

def MSE(y_pred, y):
    return np.sum(np.square(y_pred - y))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Train sets
X_0 = tf.placeholder(tf.float32, shape=[None, X_dim])
X_1 = tf.placeholder(tf.float32, shape=[None, X_dim])
X_2 = tf.placeholder(tf.float32, shape=[None, X_dim])
X_3 = tf.placeholder(tf.float32, shape=[None, X_dim])
X_4 = tf.placeholder(tf.float32, shape=[None, X_dim])

# Targets
y_0 = tf.placeholder(tf.float32, shape=[None, y_dim])
y_1 = tf.placeholder(tf.float32, shape=[None, y_dim])
y_2 = tf.placeholder(tf.float32, shape=[None, y_dim])
y_3 = tf.placeholder(tf.float32, shape=[None, y_dim])
y_4 = tf.placeholder(tf.float32, shape=[None, y_dim])

# We are generating two noise distributions
z_0 = tf.placeholder(tf.float32, shape=[None, z_dim])
z_1 = tf.placeholder(tf.float32, shape=[None, z_dim])
z_2 = tf.placeholder(tf.float32, shape=[None, z_dim])
z_3 = tf.placeholder(tf.float32, shape=[None, z_dim])
z_4 = tf.placeholder(tf.float32, shape=[None, z_dim])

# Generators
G0_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
G0_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G0_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G0_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G1_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
G1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G1_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G1_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G2_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
G2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G2_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G2_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G3_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
G3_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G3_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G3_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G4_W1 = tf.Variable(xavier_init([z_dim + y_dim, h_dim]))
G4_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G4_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G4_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

def generator_0(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    G0_h1 = tf.nn.relu(tf.matmul(inputs, G0_W1) + G0_b1)
    G0_log_prob = tf.matmul(G0_h1, G0_W2) + G0_b2
    G0_prob = tf.nn.sigmoid(G0_log_prob)
    return G0_prob

def generator_1(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    G1_h1 = tf.nn.relu(tf.matmul(inputs, G1_W1) + G1_b1)
    G1_log_prob = tf.matmul(G1_h1, G1_W2) + G1_b2
    G1_prob = tf.nn.sigmoid(G1_log_prob)
    return G1_prob

def generator_2(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    G2_h1 = tf.nn.relu(tf.matmul(inputs, G2_W1) + G2_b1)
    G2_log_prob = tf.matmul(G2_h1, G2_W2) + G2_b2
    G2_prob = tf.nn.sigmoid(G2_log_prob)
    return G2_prob

def generator_3(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    G3_h1 = tf.nn.relu(tf.matmul(inputs, G3_W1) + G3_b1)
    G3_log_prob = tf.matmul(G3_h1, G3_W2) + G3_b2
    G3_prob = tf.nn.sigmoid(G3_log_prob)
    return G3_prob

def generator_4(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    G4_h1 = tf.nn.relu(tf.matmul(inputs, G4_W1) + G4_b1)
    G4_log_prob = tf.matmul(G4_h1, G4_W2) + G4_b2
    G4_prob = tf.nn.sigmoid(G4_log_prob)
    return G4_prob


D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2_gan = tf.Variable(xavier_init([h_dim, 1]))
D_b2_gan = tf.Variable(tf.zeros(shape=[1]))
D_W2_aux = tf.Variable(xavier_init([h_dim, y_dim]))
D_b2_aux = tf.Variable(tf.zeros(shape=[y_dim]))

D0_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D0_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D0_W2_gan = tf.Variable(xavier_init([h_dim, 1]))
D0_b2_gan = tf.Variable(tf.zeros(shape=[1]))
D0_W2_aux = tf.Variable(xavier_init([h_dim, y_dim]))
D0_b2_aux = tf.Variable(tf.zeros(shape=[y_dim]))

D1_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D1_W2_gan = tf.Variable(xavier_init([h_dim, 1]))
D1_b2_gan = tf.Variable(tf.zeros(shape=[1]))
D1_W2_aux = tf.Variable(xavier_init([h_dim, y_dim]))
D1_b2_aux = tf.Variable(tf.zeros(shape=[y_dim]))

D2_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D2_W2_gan = tf.Variable(xavier_init([h_dim, 1]))
D2_b2_gan = tf.Variable(tf.zeros(shape=[1]))
D2_W2_aux = tf.Variable(xavier_init([h_dim, y_dim]))
D2_b2_aux = tf.Variable(tf.zeros(shape=[y_dim]))

D3_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D3_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D3_W2_gan = tf.Variable(xavier_init([h_dim, 1]))
D3_b2_gan = tf.Variable(tf.zeros(shape=[1]))
D3_W2_aux = tf.Variable(xavier_init([h_dim, y_dim]))
D3_b2_aux = tf.Variable(tf.zeros(shape=[y_dim]))

D4_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D4_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D4_W2_gan = tf.Variable(xavier_init([h_dim, 1]))
D4_b2_gan = tf.Variable(tf.zeros(shape=[1]))
D4_W2_aux = tf.Variable(xavier_init([h_dim, y_dim]))
D4_b2_aux = tf.Variable(tf.zeros(shape=[y_dim]))

def discriminator(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    out_aux = tf.matmul(D_h1, D_W2_aux) + D_b2_aux
    return out_aux

def discriminator_0(X):
    D0_h1 = tf.nn.relu(tf.matmul(X, D0_W1) + D0_b1)
    out_gan = tf.nn.sigmoid(tf.matmul(D0_h1, D0_W2_gan) + D0_b2_gan)
    return out_gan

def discriminator_1(X):
    D1_h1 = tf.nn.relu(tf.matmul(X, D1_W1) + D1_b1)
    out_gan = tf.nn.sigmoid(tf.matmul(D1_h1, D1_W2_gan) + D1_b2_gan)
    return out_gan

def discriminator_2(X):
    D2_h1 = tf.nn.relu(tf.matmul(X, D2_W1) + D2_b1)
    out_gan = tf.nn.sigmoid(tf.matmul(D2_h1, D2_W2_gan) + D2_b2_gan)
    return out_gan

def discriminator_3(X):
    D3_h1 = tf.nn.relu(tf.matmul(X, D3_W1) + D3_b1)
    out_gan = tf.nn.sigmoid(tf.matmul(D3_h1, D3_W2_gan) + D3_b2_gan)
    return out_gan

def discriminator_4(X):
    D4_h1 = tf.nn.relu(tf.matmul(X, D4_W1) + D4_b1)
    out_gan = tf.nn.sigmoid(tf.matmul(D4_h1, D4_W2_gan) + D4_b2_gan)
    return out_gan


# Define parameters
theta_G0 = [G0_W1, G0_W2, G0_b1, G0_b2]
theta_G1 = [G1_W1, G1_W2, G1_b1, G1_b2]
theta_G2 = [G2_W1, G2_W2, G2_b1, G2_b2]
theta_G3 = [G3_W1, G3_W2, G3_b1, G3_b2]
theta_G4 = [G4_W1, G4_W2, G4_b1, G4_b2]

theta_D  = [D_W1, D_W2_gan, D_W2_aux, D_b1, D_b2_gan, D_b2_aux]
theta_D0 = [D0_W1, D0_W2_gan, D0_W2_aux, D0_b1, D0_b2_gan, D0_b2_aux]
theta_D1 = [D1_W1, D1_W2_gan, D1_W2_aux, D1_b1, D1_b2_gan, D1_b2_aux]
theta_D2 = [D2_W1, D2_W2_gan, D2_W2_aux, D2_b1, D2_b2_gan, D2_b2_aux]
theta_D3 = [D3_W1, D3_W2_gan, D3_W2_aux, D3_b1, D3_b2_gan, D3_b2_aux]
theta_D4 = [D4_W1, D4_W2_gan, D4_W2_aux, D4_b1, D4_b2_gan, D4_b2_aux]

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def cross_entropy(logit, y):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

G0_sample = generator_0(z_0, y_0)
G1_sample = generator_1(z_1, y_1)
G2_sample = generator_2(z_2, y_2)
G3_sample = generator_3(z_3, y_3)
G4_sample = generator_4(z_4, y_4)

D0_real = discriminator_0(X_0)
D1_real = discriminator_1(X_1)
D2_real = discriminator_2(X_2)
D3_real = discriminator_3(X_3)
D4_real = discriminator_4(X_4)

D0_fake = discriminator_0(G0_sample)
D1_fake = discriminator_1(G1_sample)
D2_fake = discriminator_2(G2_sample)
D3_fake = discriminator_3(G3_sample)
D4_fake = discriminator_4(G4_sample)

C0_real  = discriminator(X_0)
C1_real  = discriminator(X_1)
C2_real  = discriminator(X_2)
C3_real  = discriminator(X_3)
C4_real  = discriminator(X_4)

C0_fake = discriminator(G0_sample)
C1_fake = discriminator(G1_sample)
C2_fake = discriminator(G2_sample)
C3_fake = discriminator(G3_sample)
C4_fake = discriminator(G4_sample)

# Cross entropy aux loss
# wait what do the reals do? We shouldn't have them right?
C0 = cross_entropy(C0_real, y_0) + cross_entropy(C0_fake, y_0)
C1 = cross_entropy(C1_real, y_1) + cross_entropy(C1_fake, y_1)
C2 = cross_entropy(C2_real, y_2) + cross_entropy(C2_fake, y_2)
C3 = cross_entropy(C3_real, y_3) + cross_entropy(C3_fake, y_3)
C4 = cross_entropy(C4_real, y_4) + cross_entropy(C4_fake, y_4)

# GAN D loss
DC0_loss = -tf.reduce_mean(tf.log(D0_real + eps) + tf.log(1. - D0_fake + eps))
DC1_loss = -tf.reduce_mean(tf.log(D1_real + eps) + tf.log(1. - D1_fake + eps))
DC2_loss = -tf.reduce_mean(tf.log(D2_real + eps) + tf.log(1. - D2_fake + eps))
DC3_loss = -tf.reduce_mean(tf.log(D3_real + eps) + tf.log(1. - D3_fake + eps))
DC4_loss = -tf.reduce_mean(tf.log(D4_real + eps) + tf.log(1. - D4_fake + eps))

# GAN's G losses
# '+ Ci_loss' penalizes class similarity
# '- Ci_loss' rewards class similarity
G0_loss = tf.reduce_mean(tf.log(D0_fake + eps))
GC0_loss = -(G0_loss - C0)

G1_loss = tf.reduce_mean(tf.log(D1_fake + eps))
GC1_loss = -(G1_loss - C1)

G2_loss = tf.reduce_mean(tf.log(D2_fake + eps))
GC2_loss = -(G2_loss - C2)

G3_loss = tf.reduce_mean(tf.log(D3_fake + eps))
GC3_loss = -(G3_loss - C3)

G4_loss = tf.reduce_mean(tf.log(D4_fake + eps))
GC4_loss = -(G4_loss - C4)

# Discriminative Classifier Loss
# Quickly goes to zero
DC_loss = -(C0 + C1 + C2 + C3 + C4)

# Discriminator loss
D_solver = (tf.train.AdamOptimizer(learning_rate=d_lr / 100)
            .minimize(DC_loss, var_list=theta_D))
D0_solver = (tf.train.AdamOptimizer(learning_rate=d_lr)
            .minimize(DC0_loss, var_list=theta_D0))
D1_solver = (tf.train.AdamOptimizer(learning_rate=d_lr)
            .minimize(DC1_loss, var_list=theta_D1))
D2_solver = (tf.train.AdamOptimizer(learning_rate=d_lr)
            .minimize(DC2_loss, var_list=theta_D2))
D3_solver = (tf.train.AdamOptimizer(learning_rate=d_lr)
            .minimize(DC3_loss, var_list=theta_D3))
D4_solver = (tf.train.AdamOptimizer(learning_rate=d_lr)
            .minimize(DC4_loss, var_list=theta_D4))

# Generator losses
G0_solver = (tf.train.AdamOptimizer(learning_rate=g_lr)
            .minimize(GC0_loss, var_list=theta_G0))
G1_solver = (tf.train.AdamOptimizer(learning_rate=g_lr)
            .minimize(GC1_loss, var_list=theta_G1))
G2_solver = (tf.train.AdamOptimizer(learning_rate=g_lr)
            .minimize(GC2_loss, var_list=theta_G2))
G3_solver = (tf.train.AdamOptimizer(learning_rate=g_lr)
            .minimize(GC3_loss, var_list=theta_G3))
G4_solver = (tf.train.AdamOptimizer(learning_rate=g_lr)
            .minimize(GC4_loss, var_list=theta_G4))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('multi-out/'):
    os.makedirs('multi-out/')

i = 0

X_train = mnist.train.images
y_train = mnist.train.labels

y_indices = np.argmax(y_train, axis=1)

# Break apart by test class
X_train_0 = X_train[y_indices == 0]
X_train_1 = X_train[y_indices == 1]
X_train_2 = X_train[y_indices == 2]
X_train_3 = X_train[y_indices == 3]
X_train_4 = X_train[y_indices == 4]

y_train_0 = y_train[y_indices == 0]
y_train_1 = y_train[y_indices == 1]
y_train_2 = y_train[y_indices == 2]
y_train_3 = y_train[y_indices == 3]
y_train_4 = y_train[y_indices == 4]

for it in range(1000000):

    b_0 = np.random.randint(0, X_train_0.shape[0], mb_size)
    b_1 = np.random.randint(0, X_train_1.shape[0], mb_size)
    b_2 = np.random.randint(0, X_train_2.shape[0], mb_size)
    b_3 = np.random.randint(0, X_train_3.shape[0], mb_size)
    b_4 = np.random.randint(0, X_train_4.shape[0], mb_size)

    X_mb_0, y_mb_0 = X_train_0[b_0], y_train_0[b_0]
    X_mb_1, y_mb_1 = X_train_1[b_1], y_train_1[b_1]
    X_mb_2, y_mb_2 = X_train_2[b_2], y_train_2[b_2]
    X_mb_3, y_mb_3 = X_train_3[b_3], y_train_3[b_3]
    X_mb_4, y_mb_4 = X_train_4[b_4], y_train_4[b_4]

    z_mb_0 = sample_z(mb_size, z_dim)
    z_mb_1 = sample_z(mb_size, z_dim)
    z_mb_2 = sample_z(mb_size, z_dim)
    z_mb_3 = sample_z(mb_size, z_dim)
    z_mb_4 = sample_z(mb_size, z_dim)

    #feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}

    # Make both zeros
    _, DC_loss_curr = sess.run(
        [D_solver, DC_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, DC0_loss_curr = sess.run(
        [D0_solver, DC0_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, DC1_loss_curr = sess.run(
        [D1_solver, DC1_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, DC2_loss_curr = sess.run(
        [D2_solver, DC2_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, DC3_loss_curr = sess.run(
        [D3_solver, DC3_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, DC4_loss_curr = sess.run(
        [D4_solver, DC4_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, GC0_loss_curr = sess.run(
        [G0_solver, GC0_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, GC1_loss_curr = sess.run(
        [G1_solver, GC1_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, GC2_loss_curr = sess.run(
        [G2_solver, GC2_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, GC3_loss_curr = sess.run(
        [G3_solver, GC3_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    _, GC4_loss_curr = sess.run(
        [G4_solver, GC4_loss],
        feed_dict={X_0: X_mb_0, y_0: y_mb_0, z_0: z_mb_0, X_1: X_mb_1, y_1: y_mb_1, z_1: z_mb_1, X_2: X_mb_2, y_2: y_mb_2, z_2: z_mb_2, X_3: X_mb_3, y_3: y_mb_3, z_3: z_mb_3, X_4: X_mb_4, y_4: y_mb_4, z_4: z_mb_4}
    )

    # Different classes
    # feed_dict={X_0: X_mb_0, X_1: X_mb_1, y_0: y_mb_0, y_1: y_mb_1, z_0: z_mb_0, z_1: z_mb_1}
    
    # Same class (1)
    # feed_dict={X_0: X_mb_1, X_1: X_mb_1, y_0: y_mb_1, y_1: y_mb_1, z_0: z_mb_1, z_1: z_mb_1}
    if it % 1000 == 0:

        n_imgs = 5
        
        # Generate samples from given indices
        c_0 = np.zeros([n_imgs, y_dim])
        c_0[range(n_imgs), 0] = 1

        c_1 = np.zeros([n_imgs, y_dim])
        c_1[range(n_imgs), 1] = 1

        c_2 = np.zeros([n_imgs, y_dim])
        c_2[range(n_imgs), 2] = 1

        c_3 = np.zeros([n_imgs, y_dim])
        c_3[range(n_imgs), 3] = 1

        c_4 = np.zeros([n_imgs, y_dim])
        c_4[range(n_imgs), 4] = 1

        samples_0 = sess.run(G0_sample, feed_dict={z_0: sample_z(n_imgs, z_dim), y_0: c_0})
        samples_1 = sess.run(G1_sample, feed_dict={z_1: sample_z(n_imgs, z_dim), y_1: c_1})
        samples_2 = sess.run(G2_sample, feed_dict={z_2: sample_z(n_imgs, z_dim), y_2: c_2})
        samples_3 = sess.run(G3_sample, feed_dict={z_3: sample_z(n_imgs, z_dim), y_3: c_3})
        samples_4 = sess.run(G4_sample, feed_dict={z_4: sample_z(n_imgs, z_dim), y_4: c_4})

        print('Iter: {}; DC_loss: {:.4}; DC0_loss: {:.4}; DC1_loss: {:.4}; GC0_loss: {:.4}; GC1_loss: {:.4}'
                      .format(it, DC_loss_curr, DC0_loss_curr, DC1_loss_curr, GC0_loss_curr, GC1_loss_curr))

        # Zero samples
        # fig = plot(samples_0)
        # plt.savefig('multi-out/{}-{}.png'
        #             .format(str(i).zfill(3), str(0)), bbox_inches='tight')
        # plt.close(fig)

        samples = [samples_0, samples_1, samples_2, samples_3, samples_4]
        fig = full_plot(samples, i)

        #y_pred = sess.run(C0_fake, feed_dict={X_0: X_train, y_0: y_train})

        # One-hot to continuous
        #y_pred_indices = np.argmax(y_pred, axis=1)
        #print(accuracy_score(y_indices, y_pred_indices))

        i += 1