# -*- coding: utf-8 -*-

import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import imageio

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Train set.
for i in range(mnist.train.num_examples):
    img = mnist.train.images[i].reshape(28, 28)
    label = mnist.train.labels[i]
    for j in range(10):
        if label[j] == 1:
            cls = j
            break
    directory_name = './images/train/' + str(cls)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, (img * 255).astype(np.uint8))

# Test set.
for i in range(mnist.test.num_examples):
    img = mnist.test.images[i].reshape(28, 28)
    label = mnist.test.labels[i]
    for j in range(10):
        if label[j] == 1:
            cls = j
            break
    directory_name = './images/test/' + str(cls)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, (img * 255).astype(np.uint8))

# Validation set.
for i in range(mnist.validation.num_examples):
    img = mnist.validation.images[i].reshape(28, 28)
    label = mnist.validation.labels[i]
    for j in range(10):
        if label[j] == 1:
            cls = j
            break
    directory_name = './images/validation/' + str(cls)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, (img * 255).astype(np.uint8))

print('OK')
