import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.framework import dtypes


@function.Defun(dtypes.float32, dtypes.float32)
def nuclear_norm_grad(x, dy):
    _, U, V = tf.svd(x, full_matrices=False, compute_uv=True)
    grad = tf.matmul(U, tf.transpose(V))
    return dy * grad


@function.Defun(dtypes.float32, grad_func=nuclear_norm_grad)
def nuclear_norm(x):
    sigma = tf.svd(x, full_matrices=False, compute_uv=False)
    norm = tf.reduce_sum(sigma)
    return norm


def TensorUnfold(A, k):
    tmp_arr = np.arange(A.get_shape().ndims)
    A = tf.transpose(A, [tmp_arr[k]] + np.delete(tmp_arr, k).tolist())
    shapeA = A.get_shape().as_list()
    A = tf.reshape(A, [shapeA[0], np.prod(shapeA[1:])])
    return A


def tensor_nuclear_norm(X, method='Tucker'):
    shapeX = X.get_shape().as_list()
    dimX = len(shapeX)
    if method == 'Tucker':
        re = [nuclear_norm(i) for i in [TensorUnfold(X, j) for j in range(dimX)]]
    elif method == 'TT':
        re = [nuclear_norm(i) for i in
              [tf.reshape(X, [np.prod(shapeX[:j]), np.prod(shapeX[j:])]) for j in range(1, dimX)]]
    elif method == 'LAF':
        re = [nuclear_norm(TensorUnfold(X, -1))]
    return tf.pack(re)


with tf.Graph().as_default() as g:
    sess = tf.InteractiveSession()
    # NN code here

    loss_norm = tf.reduce_mean(
        tf.pack([tf.reduce_mean(tensor_nuclear_norm(j, 'LAF')) for j in [W_conv1, W_conv2, W_conv3, W_fc1]]))

    # W_conv1 is the stacked W_conv1's from all NNs' first layers
