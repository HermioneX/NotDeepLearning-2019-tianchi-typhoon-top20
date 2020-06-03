# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:20:25 2019

@author: HermioneX
"""

from skimage.io import imsave
from scipy import ndimage
import numpy as np
from numba import jit
import time
import cv2
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


@jit
def match_scores(desc1, desc2):

    dist_ratio = 0.95
    desc1_size = desc1.shape
    matchscores = np.zeros((desc1_size[0]),'int')
    desc2t = desc2.T # precompute matrix transpose
    for i in range(desc1_size[0]):
        print(i)
        dotprods = np.dot(desc1[i,:],desc2t) # vector of dot products
        dotprods = 0.9999*dotprods
        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))
        
        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
    return matchscores


def match_twosided(desc1,desc2):
    """ Two-sided symmetric version of match(). """
    print('norm desc >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    print('match >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    matches_12 = match_scores(desc1,desc2)
    matches_21 = match_scores(desc2,desc1)
    
    ndx_12 = matches_12.nonzero()[0]  ## 非零下标
    
    print('remove >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
    
    return matches_12

def detect_fast_sift(image):
    fast = cv2.FastFeatureDetector_create(threshold=50,nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    kp = fast.detect(image,None)
    kp, des = sift.compute(image, kp)
    des = des.astype( np.uint8 )
    return kp, des

def detect_orb(image):
    orb = cv2.ORB_create(128)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def match_bf(img1, img2, kp1, kp2, des1, des2):
    mp = bf.match(des1, des2)
    mp = sorted(mp, key = lambda x: x.distance)
    outImg = None
    outImg = cv2.drawMatches(img1, kp1, img2, kp2, mp[:], outImg, matchColor=(0,255,0),
            flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    return outImg

def p2d(pt):
    return tuple(np.array(pt,np.uint16))

def layers(data, ch):
    data[data < th[ch]] = data_range[ch][0]
    print('Layering>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ', th[ch])
    return data

def erosion(data):
    return ndimage.binary_erosion(data).astype(data.dtype)

def good_match(img1, img2, kp1, kp2, des1, des2):
        
    print('Get locs >>>>>>>>>>>>>>>>>>>>>>>>>')
    locs1 = [(lambda x:x.pt) (x) for x in kp1]
    locs2 = [(lambda x:x.pt) (x) for x in kp2]
    
    print('match_twosided >>>>>>>>>>>>>>>>>>>>>>>>>')
    gmp = match_twosided(des1, des2)
    
    print('get src_pts >>>>>>>>>>>>>>>>>>>>>>>>>')
    src_pts = [locs1[i] for i in np.where(gmp>0)[0]]
    dst_pts = [ locs2[i] for i in gmp[gmp>0]]
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    
    disp = np.asarray(dst_pts) - np.asarray(src_pts)

    remove_ind  =  np.where(np.sum(abs(disp),axis = 1) < 200)
    src_pts = src_pts[remove_ind  ]
    dst_pts = dst_pts[remove_ind ]
    
    print('out img and lines >>>>>>>>>>>>>>>>>>>>>>>>>')
    outImg = np.hstack((img1,img2))
    dst_pts2 = np.array([[dst_pts[n][0] + 1999, dst_pts[n][1]] for n in range(dst_pts.shape[0])])
    #for i in range(dst_pts.shape[0]):
    # ind_rd = np.random.randint(1, dst_pts.shape[0], 200)
    ind_rd = np.random.randint(1, dst_pts.shape[0], 200)
    for i in ind_rd:
        cv2.line(outImg, p2d(src_pts[i]), p2d(dst_pts2[i]), (0, 0, 255), 1, 16)

    # print('out img and lines >>>>>>>>>>>>>>>>>>>>>>>>>')
    # outImg = img1.copy()
    # # dst_pts2 = np.array([[dst_pts[n][0] + 1999, dst_pts[n][1]] for n in range(dst_pts.shape[0])])
    # #for i in range(dst_pts.shape[0]):
    # ind_rd = np.random.randint(1,dst_pts.shape[0] , 100)
    # for i in ind_rd:
    #     cv2.line(outImg, p2d(src_pts[i]), p2d(dst_pts[i]), (0, 0, 255), 1, 16)

    # return outImg, src_pts,dst_pts

def get_color(data_path, head, time):
    ch_r = np.load(data_path + head + str(time) + '_Band_08.npy')
    ch_g = np.load(data_path + head + str(time) + '_Band_09.npy')
    ch_b = np.load(data_path + head + str(time) + '_Band_10.npy')
    
    ch_min = 1600
    ch_max = 4000
    
    ch_r = layers(ch_r,8)
    ch_g = layers(ch_g,9)
    ch_b = layers(ch_b,10)


    img_all = np.dstack((ch_r, ch_g, ch_b))
    img_rgb = np.array((np.array(img_all,np.float32).copy() - ch_min)*255/(ch_max - ch_min), np.uint8)
    
    return img_rgb


if __name__ == '__main__':

    st_time = time.time()

    #data_path  = 'E:\\workData\\2019.7.30_typhoon_tianchi\\'
    ##data_path  = 'E:\\workData\\2019.7.30_typhoon_tianchi\\round1_testset\\'
    
    ######################## read jpg
    data_range = {8:[2700,4010],
                   9:[1800,3990],
                   10:[2450,3780]} #8,9,10 三个通道的数据范围
    th = {8:3850,
           9:3750,
           10:3700} #8,9,10 三个通道的数据范围
    
    ########################## read npy 
    fhead = 'U_Hour_'
    root_path = 'D:\\workData\\2019.7.30_typhoon_tianchi\\round1_testset\\'
    start_ind = 213
    pred_step = 1

    img1 = get_color(root_path, fhead, start_ind - pred_step)
    img2 = get_color(root_path, fhead, start_ind)
    
    ########################## get gray
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    image1 = gray1.copy()
    image2 = gray2.copy()
    ####################  fast detect
    print('>>>>>>>>>>>> fast detecting >>>>>>>>>>>>>>>>>>>>>>>')
    kp1, des1 = detect_fast_sift(image1)
    kp2, des2 = detect_fast_sift(image2)
    #####################  sift match and draw
    #print('>>>>>>>>>>>> sift matching >>>>>>>>>>>>>>>>>>>>>>>')
    #imsave('matches_fast_sift.jpg',match_bf(image1, image2, kp1, kp2, des1, des2))
    ###################### good match and  my own plot
    print('>>>>>>>>>>>> good matching >>>>>>>>>>>>>>>>>>>>>>>')
    rst_img, srcp, desp = good_match(img1, img2, kp1, kp2, des1, des2)
    imsave('matches_good_match_'+ fhead + str(start_ind) + '_'+ str(pred_step) +'_h.jpg',rst_img)
    
    # #################### orb
    # print('>>>>>>>>>>>> orb detecting and  matching >>>>>>>>>>>>>>>>>>>>>>>')
    # kp1, des1 = detect_orb(image1)
    # kp2, des2 = detect_orb(image2)
    # imsave('matches_orb.jpg',match_bf(image1, image2, kp1, kp2, des1, des2))


    ed_time = time.time()
    print('All time:::::::::',ed_time - st_time,' seconds')







