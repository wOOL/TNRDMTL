import autograd.numpy as np
from autograd import grad
from autograd.scipy.linalg import svd, inv, sqrtm

N = 1000
P = 100

X = np.random.randn(N, P) * 0.1

norm0 = np.sum(svd(X, compute_uv=False))

norm1 = np.trace(sqrtm(np.dot(X.T, X)))

np.allclose(norm0, norm1)

np.allclose(np.dot(X, inv(sqrtm(np.dot(X.T, X)))), np.dot(X, sqrtm(inv(np.dot(X.T, X)))))

grad0 = np.dot(X, inv(sqrtm(np.dot(X.T, X))))

U, _, V = svd(X, full_matrices=False)
grad1 = np.dot(U, V)

grad2 = grad(lambda X: np.trace(sqrtm(np.dot(X.T, X))))(X)

np.allclose(grad0, grad1), np.allclose(grad1, grad2)

N = 100
P = 1000

X = np.random.randn(N, P) * 0.1

norm0 = np.sum(svd(X, compute_uv=False))

norm1 = np.trace(sqrtm(np.dot(X, X.T)))

np.allclose(norm0, norm1)

np.allclose(np.dot(X.T, inv(sqrtm(np.dot(X, X.T)))).T, np.dot(X.T, sqrtm(inv(np.dot(X, X.T)))).T)

grad0 = np.dot(X.T, inv(sqrtm(np.dot(X, X.T)))).T

U, _, V = svd(X, full_matrices=False)
grad1 = np.dot(U, V)

grad2 = grad(lambda X: np.trace(sqrtm(np.dot(X, X.T))))(X)

np.allclose(grad0, grad1), np.allclose(grad1, grad2)
