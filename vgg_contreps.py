import numpy as np

def vgg_contreps(X=None,*args,**kwargs):

    # vgg_contreps  Contraction with epsilon tensor.

    # B = vgg_contreps(A) is tensor obtained by contraction of A with epsilon tensor.
# However, it works only if the argument and result fit to matrices, in particular:

    # - if A is row or column 3-vector ...  B = [A]_x
# - if A is skew-symmetric 3-by-3 matrix ... B is row 3-vector such that A = [B]_x
# - if A is skew-symmetric 4-by-4 matrix ... then A can be interpreted as a 3D line Pluecker matrix
#                                               skew-symmetric 4-by-4 B as its dual Pluecker matrix.
# - if A is row 2-vector ... B = [0 1; -1 0]*A', i.e., A*B=eye(2)
# - if A is column 2-vector ... B = A'*[0 1; -1 0], i.e., B*A=eye(2)

    # It is vgg_contreps(vgg_contreps(A)) = A.

    # werner@robots.ox.ac.uk, Oct 2001

    # we know in advance that the input is a column 3-vector
    return np.array([[0,X[2],- X[1]], [- X[2],0,X[0]], [X[1],- X[0],0]])
