from scipy import linalg
import numpy as np

# Initialize M
M = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
print("The matrix M is:\n", M)

U, s, Vh = linalg.svd(M, full_matrices = False)

print("U is :\n", U)
print("Singular matrix is:\n", s)
print("V transpose is:\n", Vh)

M_transpose = M.T
print("M_transpose * M = \n", np.matmul(M_transpose, M))
w, v = linalg.eigh(np.matmul(M_transpose, M))
idx = np.argsort(-w)
w = w[idx]
v = v[:,idx]
print("Evals are:\n", w)
print("Evecs are:\n", v)
