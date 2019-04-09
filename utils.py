import mnist

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2


def ones_like(x, size):
    return torch.ones(size, dtype=x.dtype, layout=x.layout, device=x.device)


def zeros_like(x, size):
    return torch.zeros(size, dtype=x.dtype, layout=x.layout, device=x.device)


def eye_like(x, size):
    return torch.eye(size, dtype=x.dtype, layout=x.layout, device=x.device)


def bmat(l):
    return torch.cat([torch.cat(x, dim=1) for x in l], dim=0)


def e(i, n):
    x = np.zeros(n)
    x[i] = 1.
    return x


def one_hot(y, n):
    N = y.size
    Y = np.zeros((N, n))
    Y[np.arange(N), y] = 1
    return Y


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_features(x_train, y_train, nclusters=10):
    centers = []
    km = KMeans(n_clusters=nclusters, n_jobs=12)
    for d in np.arange(10):
        km.fit(x_train[y_train.argmax(axis=1) == d])
        centers.append(km.cluster_centers_.copy())
    centers = np.concatenate(centers)
    return centers.T


def deskew(image, image_shape, negated=False):
    # https://github.com/vsvinayak/mnist-helper
    """
    The MIT License (MIT)

    Copyright (c) 2015 Vinayak V S

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    """
    This method deskwes an image using moments
    :param image: a numpy nd array input image
    :param image_shape: a tuple denoting the image`s shape
    :param negated: a boolean flag telling  whether the input image is a negated one
    :returns: a numpy nd array deskewd image
    """

    # negate the image
    if not negated:
        image = 255 - image

    # calculate the moments of the image
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()

    # caclulating the skew
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * image_shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(image, M, image_shape,
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    return img


def get_mnist(num_train):
    x_train, y_train, x_test, y_test = mnist.load()
    x_train, y_train = shuffle(x_train, y_train)
    x_train, y_train = x_train[:num_train], y_train[:num_train]
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=.3)
    return x_train, y_train, x_val, y_val, x_test, y_test
