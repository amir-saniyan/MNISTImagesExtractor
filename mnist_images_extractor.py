# -*- coding: utf-8 -*-

# MNIST Images Extractor
# Python code for extracting MNIST dataset images.

# MNIST Dataset:
# http://yann.lecun.com/exdb/mnist/

# Repository:
# https://github.com/amir-saniyan/MNISTImagesExtractor

import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import imageio

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

# Train set.
for i in range(mnist.train.num_examples):
    image = mnist.train.images[i].reshape(28, 28)
    label = mnist.train.labels[i]
    directory_name = './images/train/' + str(label)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, (image * 255).astype(np.uint8))

# Test set.
for i in range(mnist.test.num_examples):
    image = mnist.test.images[i].reshape(28, 28)
    label = mnist.test.labels[i]
    directory_name = './images/test/' + str(label)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, (image * 255).astype(np.uint8))

# Validation set.
for i in range(mnist.validation.num_examples):
    image = mnist.validation.images[i].reshape(28, 28)
    label = mnist.validation.labels[i]
    directory_name = './images/validation/' + str(label)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, (image * 255).astype(np.uint8))

print('OK')
