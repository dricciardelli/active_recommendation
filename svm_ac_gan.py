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

# to use all of mnist and all generated images as training data
X_train = np.load("mnist_a_train_img.npy")
y_train = np.load("mnist_a_train_label.npy")

# to use only generated images as training data
#X_train = np.load("generated_img.npy")
#y_train = np.load("generated_label.npy")

y_train = [np.argmax(one_hot) for one_hot in list(y_train)]

# to use original mnist images as training data
#X_train = mnist.train.images
#y_train = [np.argmax(one_hot) for one_hot in list(mnist.train.labels)]
X_test = mnist.test.images
y_test = [np.argmax(one_hot) for one_hot in list(mnist.test.labels)]

print(X_train.shape)

# Train
#if os.path.exists('svm_basic.sav'):
if load_model:
    svm_basic = pickle.load(open('svm_basic.sav', 'rb'))
else:
	svm_basic = SVC()
	print ("Training svm ... ")
	svm_basic.fit(X_train, y_train)
	pickle.dump(svm_basic, open("svm_augment.sav", 'wb'))
	print ("Trained and saved!")

# Predict
z_train = svm_basic.predict(X_train[:1000])
print(z_train.shape)
print(np.array(y_train[:1000]).shape)
print (("Training Accuracy: {0:.0f}%").format(accuracy_score(y_train[:1000], z_train)*100))

z_test = svm_basic.predict(X_test[:1000])
print(z_test.shape)
print(np.array(y_test[:1000]).shape)
print (("Test Accuracy: {0:.0f}%").format(accuracy_score(y_test[:1000], z_test)*100))