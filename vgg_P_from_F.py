import numpy as np
from vgg_contreps import vgg_contreps

    #P = vgg_P_from_F(F)  Compute cameras from fundamental matrix.
#   F has size (3,3), P has size (3,4).

    #   If x2'*F*x1 = 0 for any pair of image points x1 and x2,
#   then the camera matrices of the image pair are
#   P1 = eye(3,4) and P2 = vgg_P_from_F(F), up to a scene homography.

    # Tomas Werner, Oct 2001


def vgg_P_from_F(F=None,*args,**kwargs):

    U,S,V=np.linalg.svd(F)
# vgg_P_from_F.m:12
    e=U[:,2]
    t=-vgg_contreps(e)
# vgg_P_from_F.m:13
    return np.concatenate((t.dot(F), np.expand_dims(e,-1)), axis=1)
