import cv2
import numpy as np
import scipy.spatial

leftname = 'images/left.png'
rightname = 'images/right.png'

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

#print (do_sift(leftname))
#print (do_sift(rightname))
lkp, ldes = do_sift(leftname)
#print (kp[0], des.shape)
rkp, rdes = do_sift(rightname)
#print (kp[0], des.shape)
print(find_correspondence(ldes, rdes))
