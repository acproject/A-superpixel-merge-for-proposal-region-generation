#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:11:39 2018

@author: ZSQ
"""

import cv2
import copy
import RPG.multi_stage_merge as multi_stage_merge
###########################start coding###########################
if __name__ == '__main__':

    rgb = cv2.imread('.\images\persons.png')
    
    
    MT_merge = multi_stage_merge.multistage_merge(numSegments = 40, Patch_size = 32, K_num = 2)
    
    superpixels = MT_merge.rgb2superpixel(rgb, 40)
    contours = MT_merge.superpixels2contours(superpixels)
    rgb_contours_copy0 = copy.deepcopy(rgb)
    for j in range(len(contours)):
        rgb_contours_copy0 = cv2.drawContours(rgb_contours_copy0, contours[j], -1, (0,255,0), 2)
    cv2.imwrite('mul_stage_merge/original_superpixel.png', rgb_contours_copy0)
    
    rgb, pre_coutours = MT_merge.image2pre_merge_coutours(rgb)
    
    # imshow pre_merge results
    rgb_contours_copy = copy.deepcopy(rgb)
    for j in range(len(pre_coutours)):
        if j%2 ==0:
            rgb_contours_copy = cv2.drawContours(rgb_contours_copy, pre_coutours[j], -1, (0,255,0), 2)
        else:
            rgb_contours_copy = cv2.drawContours(rgb_contours_copy, pre_coutours[j], -1, (255,0,0), 2)

    rgb_rect_copy = copy.deepcopy(rgb)
    rects_copy = MT_merge.contours2rects(pre_coutours)
    for j in range(len(rects_copy)):
        x,y,w,h = rects_copy[j]
        rgb_rect_copy = cv2.rectangle(rgb_rect_copy,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imwrite('mul_stage_merge/rgb_contours_copy.png', rgb_contours_copy)
    cv2.imwrite('mul_stage_merge/rgb_rect_copy.png', rgb_rect_copy)