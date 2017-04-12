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

d_steps = 3

restore = False


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
    plt.savefig('out2/sample_{}'.format(str(itr).zfill(3)) , dpi=600)
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
saver = tf.train.Saver()

if not os.path.exists('out/'):
    os.makedirs('out/')

itr = 0

X_train = mnist.train.images
y_train = mnist.train.labels

X_test = mnist.test.images
y_test = mnist.test.labels

y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

if (restore):
    aver.restore(sess, "out/model.ckpt")
    print("Model restored.")

    n_imgs = 1000

    # Pick all numbers to generate samples from
    c_list = [np.zeros([n_imgs, y_dim]) for _ in range(10)]
    for i in range(10):
        c_list[i][range(n_imgs), i] = 1

    mnist_augmented_train_images = mnist.train.images
    mnist_augmented_train_labels = mnist.train.labels
    sample = sess.run(G_sample, feed_dict={z: sample_z(n_imgs, z_dim), y: c_list[0]})
    generated_images = np.array(sample)
    generated_labels = np.array(np.array(c_list[i]))
    for i in range(10):
        sample = sess.run(G_sample, feed_dict={z: sample_z(n_imgs, z_dim), y: c_list[i]})
        mnist_augmented_train_images = np.concatenate((mnist_augmented_train_images, sample), axis=0)
        mnist_augmented_train_labels = np.concatenate((mnist_augmented_train_labels, np.array(c_list[i])), axis=0)
        if (i != 0):
            generated_images = np.concatenate((generated_images, sample), axis=0)
            generated_labels = np.concatenate((generated_labels, np.array(c_list[i])), axis=0)

    np.save('mnist_a_train_img.npy', mnist_augmented_train_images)
    np.save('mnist_a_train_label.npy', mnist_augmented_train_labels)
    np.save('generated_img.npy', generated_images)
    np.save('generated_label.npy', generated_labels)

else:
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

            print('Iter: {}; DC_loss: {:.4}; GC_loss: {:.4}'
                  .format(it, DC_loss_curr, GC_loss_curr))

            train_preds = sess.run(C_real, feed_dict={X: X_train})
            pred_train_indices = np.argmax(train_preds, axis=1)

            test_preds = sess.run(C_real, feed_dict={X: X_test})
            pred_test_indices = np.argmax(test_preds, axis=1)

            print('Train Accuracy: {}'.format(accuracy_score(y_train_indices, pred_train_indices)))
            print('Test Accuracy: {}'.format(accuracy_score(y_test_indices, pred_test_indices)))

            full_plot(samples_list, itr)
            save_path = saver.save(sess, "out/model.ckpt")
            print("Model saved in file: %s" % save_path)

            itr += 1