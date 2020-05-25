#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:11:39 2018

@author: ZSQ
"""
import RPG.Region_Proposal as region_proposal
import numpy as np
import cv2
import copy

###########################start coding###########################

if __name__ == '__main__':

    print('start coding .............')
    rgb = cv2.imread('.\images\3.png')
    # numSegments --initial superpixel num 
    # Patch_size ---image nomalizational patch size of computing adjacent region
    # Knum -- popasal region num
    RP = region_proposal.region_proposal(numSegments = 50, Patch_size = 40,  K_num = 15)
    
    # rgb2superpixels
    superpixels = RP.rgb2superpixel(rgb)
    contours = RP.superpixels2contours(superpixels)
    contours_copy = copy.copy(contours)
    # merge
    pre_contours = RP.pre_region_merge(rgb, rgb[:,:,1], contours_copy)
    
    # plot contours
    rgb1 = rgb.copy()
    for j in range(len(pre_contours)):
        rgb1 = cv2.drawContours(rgb1, pre_contours[j], -1, (0,255,0), 2)

    rgb2 = rgb.copy()
    for j in range(len(contours)):
        rgb2 = cv2.drawContours(rgb2, contours[j], -1, (0,255,0), 2)
        
    #stackimg = np.stack((rgb1[:,:,1], rgb2[:,:,1]), axis = 1)
    # imwrite results
    cv2.imshow('rgb1', rgb1)
    cv2.imshow('rgb2', rgb2)
    cv2.waitKey(0)
    cv2.imwrite('./results/original3.png', rgb)
    cv2.imwrite('./results/original3_superpixel.png', rgb1)
    cv2.imwrite('./results/merge3.png', rgb2)
    