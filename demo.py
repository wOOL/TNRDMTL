import tensorflow as tf
import numpy as np
from scipy.linalg import svd


def nuclear_norm(X):
    return np.sum(svd(X, compute_uv=False))


def nuclear_norm_grad(X):
    U, _, V = svd(X, full_matrices=False)
    return np.dot(U, V)


def nuclear_norm_tf(X):
    return tf.py_func(nuclear_norm, [X], [tf.float32], name='nuclear_norm')[0]


@tf.RegisterGradient("nuclear_norm")
def nuclear_norm_grad_tf(op, grad):
    X = op.inputs[0]
    return grad * tf.py_func(nuclear_norm_grad, [X], [tf.float32], name='nuclear_norm_grad')[0]


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
        re = [nuclear_norm_tf(i) for i in [TensorUnfold(X, j) for j in range(dimX)]]
    elif method == 'TT':
        re = [nuclear_norm_tf(i) for i in
              [tf.reshape(X, [np.prod(shapeX[:j]), np.prod(shapeX[j:])]) for j in range(1, dimX)]]
    elif method == 'SVD':
        re = [nuclear_norm_tf(TensorUnfold(X, -1))]
    return tf.pack(re)


with tf.Graph().as_default() as g:
    with g.gradient_override_map({"PyFunc": "nuclear_norm"}):
        # NN code here

        loss_norm = tf.reduce_mean(
            tf.pack([tf.reduce_mean(tensor_nuclear_norm(j, 'SVD')) for j in [W_conv1, W_conv2, W_conv3, W_fc1]]))

        # W_conv1 is the stacked W_conv1's from all NNs' first layers
