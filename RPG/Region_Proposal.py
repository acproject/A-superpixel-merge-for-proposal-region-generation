# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:04:59 2018

@author: ZSQ
"""

## import the necessary packages
from PIL import Image 
import scipy.io as sio  
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy

# define rp class
class region_proposal(object):
    def __init__(self, numSegments = 50, Patch_size = 40,  K_num = 20):
        self.numSegments = numSegments
        self.Patch_size = Patch_size
        self.K_num = K_num

    # covert the image data format
    def mat2rgb(self, image):
        img = image['image']
        # input2rgb
        img = np.floor((img - img.min())*255/(img.max() - img.min()))
        img = np.uint8(img)
        rgb = np.ones((img.shape[0], img.shape[1], 3), dtype= np.uint8)
        rgb[:,:,0] = img
        rgb[:,:,1] = img
        rgb[:,:,2] = img
        
        return rgb
    
    # SLIC superpixel   
    def rgb2superpixel(self, image):
        image = img_as_float(image)
        segments = slic(image, n_segments = self.numSegments, sigma = 5, compactness=40)
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
    def compute_adjac_table_dist_tabel(self, image, rects, sum_contours, patch_size):
        Adjacent_table = {}
        dist_table = {}
        normalize_img =  (image - image.min())/(image.max() - image.min())
        for i in range(len(sum_contours)):
            draw_board = np.zeros(image.shape, np.uint8)
            # get image patch feature
            [x,y,w,h] = rects[i]
            ref_patch = normalize_img[y:(y + h),x:(x + w)]
    #        ref_patch = normalize_img[rects[i][0]:(rects[i][0] + rects[i][2]), rects[i][1]:(rects[i][1] + rects[i][3])] 
            resize_ref_patch = cv2.resize(ref_patch, (patch_size,patch_size),interpolation=cv2.INTER_CUBIC)
            index_list = []
            dist_list = []
            for j in range(len(sum_contours)):
                draw_board = cv2.drawContours(draw_board, sum_contours[i], -1, (255,255,255), -1)
                draw_board = cv2.drawContours(draw_board, sum_contours[j], -1, (255,255,255), -1)
                
                # get single Contours
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))  
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
            tmp_dict = {}
            if len(index_list) ==0:
                if i!=0:
                    index_list.append(i-1)
                else:
                    index_list.append(i)
                
            tmp_dict[str(i)] = index_list
            Adjacent_table.update(tmp_dict)
            
            # get dist table
            tmp_dict = {}
            if len(dist_list) ==0:
                dist_list.append(0)
            tmp_dict[str(i)] = dist_list
            dist_table.update(tmp_dict)
            
        return Adjacent_table, dist_table
    
    # compute adjact region similarity list
    def compute_most_similar_Adjacent_tabel(self, Adjacent_table, dist_table):
        adjact_list = []
        dist_list = []
        for i in range(len(Adjacent_table)):
    #        Adjacent_dist_table = np.zeros((len(Adjacent_table), 2), np.float)
            index = dist_table[str(i)].index(min(dist_table[str(i)]))
            sum_index = Adjacent_table[str(i)][index]
            min_value = min(dist_table[str(i)])
            adjact_list.append(sum_index)
            dist_list.append(min_value)
        return adjact_list, dist_list
    
    # choose the most similar region to merge
    def most_similar_Adjacent_region_merge(self, image, contours, adjact_list, dist_list):
        draw_board = np.zeros(image.shape[:-1], np.uint8)
        sortlist = sorted(dist_list)
        middle_value = sortlist[round(len(sortlist)/10)]
        #index = dist_list.index(min(dist_list))
        index = dist_list.index(middle_value)
        draw_board = cv2.drawContours(draw_board, contours[index], -1, (255,255,255), -1)
        draw_board = cv2.drawContours(draw_board, contours[adjact_list[index]], -1, (255,255,255), -1)
        draw_board, contour, hierarchy = cv2.findContours(draw_board, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        if adjact_list[index] > index:
            contours.pop(adjact_list[index])
            contours.pop(index)
        else:
            contours.pop(index)
            contours.pop(adjact_list[index])
        contours.append(contour)
        return contours
    
    # Merge similar region until remaining regions is less than K_num
    def pre_region_merge(self, rgb, image, contours):
        while True:
            rects = self.contours2rects(contours)
            adjac_table, dist_table = self.compute_adjac_table_dist_tabel(image, rects, contours, self.Patch_size)
            adjact_list, dist_list = self.compute_most_similar_Adjacent_tabel(adjac_table, dist_table)    
            contours = self.most_similar_Adjacent_region_merge(rgb, contours, adjact_list, dist_list)
            
            print('the number of superpixel region' + ' ' + str(len(contours)))
            if len(contours) < self.K_num:
                break
        return contours
    
if __name__ == '__main__':
    pass