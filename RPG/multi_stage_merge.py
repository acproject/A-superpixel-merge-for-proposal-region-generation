# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 19:22:45 2018

@author: ZSQ
"""

## import the necessary packages
from PIL import Image 
import scipy.io as sio  
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from skimage.feature import local_binary_pattern
from skimage.feature import greycoprops
from skimage.feature import greycomatrix
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy

# SLIC superpixel 
class multistage_merge(object):
    def __init__(self, numSegments = 40, Patch_size = 32, K_num = 2):
        self.numSegments = numSegments
        self.Patch_size = Patch_size
        self.K_num = K_num
        
    def rgb2superpixel(self, image, n_segments):
        image = img_as_float(image)
        segments = slic(image, n_segments = self.numSegments, sigma = 5, compactness=50)
        return segments
    
    # Get regions rects and contours
    def superpixels2contours(self, superpixels):
        values = np.unique(superpixels)
        sum_contours = []
        for index in range(len(values)):
            L3 = np.zeros(superpixels.shape, np.uint8)
            L3[np.where(superpixels==values[index])] = 1
            # extract cotours
            ret,thresh = cv2.threshold(L3, 0, 1, 0)
            image, contours, hierarchy = cv2.findContours(L3, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            sum_contours.append(contours)
        return sum_contours
    
    def contours2rects(self, contours):
        rects = []
        for i in range(len(contours)):
            rect =[]
            x,y,w,h = cv2.boundingRect(contours[i][0])
            rect = [x,y,w,h]
            rects.append(rect)
        return rects
    
    ## Get adjacent_table and dist_table
    def compute_adjac_table_dist_tabel(self, rgb, image, rects, sum_contours, patch_size):
        Adjacent_table = {}
        dist_table = {}
        normalize_img =  (image- image.min())/(image.max() - image.min())
    #    normalize_img = image
        for i in range(len(sum_contours)):
            draw_board = np.zeros(image.shape, np.uint8)
            # get image patch feature
            [x,y,w,h] = rects[i]
            ref_patch = normalize_img[y:(y + h),x:(x + w)]
            resize_ref_patch = cv2.resize(ref_patch, (patch_size,patch_size),interpolation=cv2.INTER_CUBIC)
            index_list = []
            dist_list = []
            for j in range(len(sum_contours)):
                draw_board = cv2.drawContours(draw_board, sum_contours[i], -1, (255,255,255), -1)
                draw_board = cv2.drawContours(draw_board, sum_contours[j], -1, (255,255,255), -1)
                
                # get single Contours
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))  
                erode = cv2.erode(draw_board, kernel)
                erode, contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                
                if i!= j and len(contours) == 1:
                    index_list.append(j)
                    [x,y,w,h] = rects[j]
                    query_patch = normalize_img[y:(y + h), x:(x + w)] 
                    resize_query_patch = cv2.resize(query_patch, (patch_size,patch_size),interpolation=cv2.INTER_CUBIC)
                    
                    # compute distance between ref_path an query_patch 
                    Eucl_diff = abs(resize_query_patch - resize_ref_patch)
                    Eucl_dist = Eucl_diff.sum()/(patch_size*patch_size)
                    dist_list.append(Eucl_dist)
                draw_board = np.zeros(image.shape, np.uint8)
                
            # get Ajacent table
            tmp_dict = {}
            tmp_dict[str(i)] = index_list
            Adjacent_table.update(tmp_dict)
            
            # get dist table
            tmp_dict = {}
            tmp_dict[str(i)] = dist_list
            dist_table.update(tmp_dict)
            
        return Adjacent_table, dist_table
    
    
    def compute_most_similar_Adjacent_table(self, Adjacent_table, dist_table):
        adjact_list = []
        dist_list = []
        
        for i in range(len(Adjacent_table)):
    
            if len(Adjacent_table[str(i)]) == 0:
                adjact_list.append(i)
                dist_list.append(0)
            else:
                index = round(len(Adjacent_table[str(i)])//1.5)
    #            index = dist_table[str(i)].index(min(dist_table[str(i)]))
                sum_index = Adjacent_table[str(i)][index]
                min_value = min(dist_table[str(i)])
                adjact_list.append(sum_index)
                dist_list.append(min_value)
        return adjact_list, dist_list
            
    def most_similar_Adjacent_region_merge(self, rgb, image, contours, adjact_list, dist_list):
        draw_board = np.zeros(rgb.shape[:-1], np.uint8)
        # loop merge adjact region
        filter_list = []
        ref_list = []
        flag1 = 0
        flag2 = 0
        for i in range(len(adjact_list)):
            tmp = [i, adjact_list[i]]
            if i in ref_list:
                flag1 = 1
            else:
                ref_list.append(i)
                
            if adjact_list[i] in ref_list:
                flag2 = 1
            else:
                ref_list.append(adjact_list[i])
            
            if flag1 ==1 or flag2 ==1:
                flag1 = 0 
                flag2 = 0
            if flag1 ==1 and flag2 ==1:
                continue
            else:
                filter_list.append(tmp)
                
        contours_tmp = []     
        for index in range(len(filter_list)):
            draw_board = cv2.drawContours(draw_board, contours[filter_list[index][0]], -1, (255,255,255), -1)
            draw_board = cv2.drawContours(draw_board, contours[filter_list[index][1]], -1, (255,255,255), -1)
            draw_board, contour, hierarchy = cv2.findContours(draw_board, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            draw_board = np.zeros(rgb.shape[:-1], np.uint8)
            
            x,y,w,h = cv2.boundingRect(contour[0])
            contours_tmp.append(contour)
        return contours_tmp
    
    def pre_region_merge(self, rgb, image, contours, Patch_size):
    #    for i in range(K_num):
        rects = self.contours2rects(contours)
        adjac_table, dist_table = self.compute_adjac_table_dist_tabel(rgb, image, rects, contours, Patch_size)
        adjact_list, dist_list = self.compute_most_similar_Adjacent_table(adjac_table, dist_table)    
        contours = self.most_similar_Adjacent_region_merge(rgb, image, contours, adjact_list, dist_list)
    
        return contours
    
    def image2pre_merge_coutours(self, rgb):
            
        # pre_merge
        superpixels = self.rgb2superpixel(rgb, self.numSegments)
        contours = self.superpixels2contours(superpixels)
        
        rgb_contours_copy0 = copy.deepcopy(rgb)
        for j in range(len(contours)):
            rgb_contours_copy0 = cv2.drawContours(rgb_contours_copy0, contours[j], -1, (0,255,0), 2)
        
        #cv2.imwrite('..\..\mul_stage_merge/original_superpixel.png', rgb_contours_copy0)
    
    
        for i in range(self.K_num):
            contours = self.pre_region_merge(rgb, rgb[:,:,1], contours, self.Patch_size)
            
        return rgb, contours