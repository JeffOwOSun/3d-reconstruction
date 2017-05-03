import cv2
import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt

leftname = 'images/left.jpeg'
rightname = 'images/right.jpeg'

EPSILON = 1e-12

def do_sift(imgname):
    img = cv2.imread(imgname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def l2_distance(a, b):
    return scipy.spatial.distance.euclidean(a, b)

def find_correspondence(left, right, num_return=8):
    score_matrix = np.zeros((len(left), len(right)))
    for row, a in enumerate(list(left)):
        for col, b in enumerate(list(right)):
            #print (row, col)
            score_matrix[row][col] = l2_distance(a,b)
    #the best match to each side
    lefts = np.argmin(score_matrix, axis=1)
    rights = np.argmin(score_matrix, axis=0)
    #the scores for the best match
    left_scores = np.amin(score_matrix, axis=1)

    pairs = []
    for l in xrange(len(lefts)):
        #best match of l from left
        r = lefts[l]
        #if l is the best match of r
        if (rights[r] == l):
            pairs.append((l, r, left_scores[l]))
    return sorted(pairs, key=lambda x: x[-1])[:num_return]

def run8point(pts1, pts2):
    #compute centers for normalizations
    c1, c2 = np.mean(pts1, 0), np.mean(pts2, 0)
    #compute scales
    scale1 = (2**.5)/np.mean([l2_distance(x,c1) for x in pts1])
    scale2 = (2**.5)/np.mean([l2_distance(x,c2) for x in pts2])
    #normalize
    normpts1 = (pts1 - c1)*scale1
    normpts2 = (pts2 - c2)*scale2
    #matrix A
    A = np.zeros((9,9), np.float32)
    for (x1,y1),(x2,y2) in zip(normpts1, normpts2):
        r = np.expand_dims(np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1],
                np.float32), -1)
        A += r*r.T

    #print("py A:", A)

    #eigen value
    W, V = np.linalg.eig(A)
    idx = W.argsort()[::-1]
    W = W[idx]
    V = V[:,idx]


    #print("py W,V:", W,V)

    #take last column of V as solution
    F0 = np.reshape(V[:,-1], [3,3])
    #print("py F0", F0)

    #U, w, Vt = np.linalg.svd(F0)
    w, U, Vt = cv2.SVDecomp(F0)
    w[2] = .0
    #print("U: ",U)
    #print("w: ",w)
    #print("Vt: ",Vt)
    F0 = np.matmul(np.matmul(U,np.diag(np.reshape(w,[-1]))),Vt)
    #print("py F0", F0)


    #construct inverse transformation
    T1 = np.reshape(np.array([scale1, 0, -scale1*c1[0], 0, scale1, -scale1*c1[0], 0, 0, 1],
            np.float32), [3,3])
    T2 = np.reshape(np.array([scale2, 0, -scale2*c2[0], 0, scale2, -scale2*c2[0], 0, 0, 1],
            np.float32), [3,3])
    F0 = np.matmul(np.matmul(T2.T,F0),T1)

    if abs(F0[2,2]) > EPSILON:
        F0 *= 1./F0[2,2]

    return F0


def main():
    img1 = cv2.imread(leftname,0)  #queryimage # left image
    img2 = cv2.imread(rightname,0) #trainimage # right image
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # now do best matchings
    pts1 = []
    pts2 = []
    # FLANN parameters
    correspondences = find_correspondence(des1, des2, 50)

    for lidx, ridx, distance in correspondences:
        pts2.append(kp2[ridx].pt)
        pts1.append(kp1[lidx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #print(pts1, pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    # We select only inlier points
    #pts1 = pts1[mask.ravel()==1]
    #pts2 = pts2[mask.ravel()==1]

    def drawlines(img1,img2,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

def sample():
    img1 = cv2.imread(leftname,0)  #queryimage # left image
    img2 = cv2.imread(rightname,0) #trainimage # right image
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # now do best matchings
    pts1 = []
    pts2 = []
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    correspondences = find_correspondence(des1, des2, 50)

    for lidx, ridx, distance in correspondences:
        pts2.append(kp2[ridx].pt)
        pts1.append(kp1[lidx].pt)

    #import pdb;pdb.set_trace();
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F0, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
    F = run8point(pts1, pts2)
    #print(F,F0)
    # We select only inlier points
    #pts1 = pts1[mask.ravel()==1]
    #pts2 = pts2[mask.ravel()==1]

    def drawlines(img1,img2,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()



if __name__ == "__main__":
    #main()
    sample()
