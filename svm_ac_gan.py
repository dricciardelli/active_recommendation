import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

load_model = False
train_amts = [100, 500, 700, 1000, 2000, 3000]

# to use all of mnist and all generated images as training data
#X_train = np.load("mnist_a_train_img.npy")
#y_train = np.load("mnist_a_train_label.npy")

# to use random subset of augmeted images as training
#idx = np.random.choice(np.arange(len(X_train)), 5000, replace=False)
#X_train = X_train[idx]
#y_train = y_train[idx]

X_test = mnist.test.images
y_test = [np.argmax(one_hot) for one_hot in list(mnist.test.labels)]
X_mnist = mnist.train.images
y_mnist = mnist.train.labels

#to use only generated images as training data
for train_amt in train_amts:
    X_gen = np.load("generated_img.npy")
    y_gen = np.load("generated_label.npy")

    idx = np.random.choice(np.arange(len(X_mnist)), train_amt, replace=False)
    X_train_m = X_mnist[idx]
    y_train_m = y_mnist[idx]

    X_train_all = np.concatenate((X_gen, X_train_m), axis=0)
    y_train_all = np.concatenate((y_gen, y_train_m), axis=0)
    y_train_all = [np.argmax(one_hot) for one_hot in list(y_train_all)]

    idx_baseline = np.random.choice(np.arange(len(X_mnist)), train_amt + 1000, replace=False)
    X_train_baseline = X_mnist[idx_baseline]
    y_train_baseline = y_mnist[idx_baseline]
    y_train_baseline = [np.argmax(one_hot) for one_hot in list(y_train_baseline)]


    print(X_train.shape)

    # Train
    #if os.path.exists('svm_basic.sav'):
    if load_model:
        svm_basic = pickle.load(open('svm_basic.sav', 'rb'))
    else:
    	svm_baseline = SVC()
        svm_baseline.fit(X_train_baseline, y_train_baseline)

        svm = SVC()
        svm.fit(X_train_all, y_train_all)
    	# pickle.dump(svm_basic, open("svm_augment.sav", 'wb'))
    	# print ("Trained and saved!")

    # Predict
    z_train_baseline = svm_baseline.predict(X_train_baseline)
    z_test_baseline = svm_baseline.predict(X_test[:1000])
    print (("Baseline Training Accuracy: {0:.0f}%").format(accuracy_score(y_train_baseline, z_train_baseline)*100))
    print (("Baseline Test Accuracy: {0:.0f}%").format(accuracy_score(y_test[:1000], z_test_baseline)*100))


    z_train = svm.predict(X_train_all)
    z_test = svm.predict(X_test[:1000])
    print (("Baseline Training Accuracy: {0:.0f}%").format(accuracy_score(y_train_all, z_train)*100))
    print (("Test Accuracy: {0:.0f}%").format(accuracy_score(y_test[:1000], z_test)*100))

