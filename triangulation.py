import numpy as np
    #Find Point in 3D space from given x1,P1,x2,P2
#       [x1] * P1 * Xw = 0
#       [x2] * P2 * Xw = 0
#  Let       [A]*[Xw;Xw] = [0;0]
#  Find [U,D,V] = SVD(A), Xw is last column of V
EPSILON=1e-12
def triangulation(x1=None,P1=None,x2=None,P2=None,*args,**kwargs):
    # Argument
#   x1,x2 = Point in 2D space [x1,...,xn ; y1,...,yn ; 1,...,1]
#   P1,P2 = Projection Transformation Matrix 3x4
#   F = Fundamental Matrix
#   Xw = Point in 3D space [X1,...,Xn ; Y1,...,Yn ; Z1,...,Zn ; 1,...,1]
    x11=x1
# triangulation.m:12
    x22=x2
    Xw = np.zeros((4,x11.shape[1]), np.float64)
# triangulation.m:13
    for i in xrange(x11.shape[1]):
        #Select point
        sx1=x11[:,i]
# triangulation.m:16
        sx2=x22[:,i]
# triangulation.m:17
        A1=sx1[0]*P1[2,:] - P1[0,:]
# triangulation.m:18
        A2=sx1[1]*P1[2,:] - P1[1,:]
# triangulation.m:1*
        A3=sx2[0]*P2[2,:] - P2[0,:]
# triangulation.m:2*
        A4=sx2[1]*P2[2,:] - P2[1,:]
# triangulation.m:21
        A=np.stack((A1,A2,A3,A4), axis=0)
# triangulation.m:23
        U,D,V=np.linalg.svd(A)
# triangulation.m:24
        X_temp=V[:,3]
# triangulation.m:26
        X_temp=X_temp / (X_temp[3]+EPSILON)
# triangulation.m:27
        Xw[:,i]=X_temp
# triangulation.m:28
    return Xw

