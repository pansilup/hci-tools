# -*- coding: utf-8 -*-
"""
Author         : pansilup
Data           : Feb-09-2022
What this does : Given a set of paragraphs and images in a document,
                 derive the optimum two column layout using combinatorial optimization

setup : 
!pip install measure
!pip install imageio
!pip install scikit-image
!pip install more-itertools
!pip install pulp
"""
import numpy as np
import scipy.ndimage
from imageio import imread
import os
import cv2
import pickle
import argparse
import pulp
from itertools import product, permutations, combinations
import random

#killer-layouts in two column format

#inputs-start--------------------------------------------
in_objs_h = [400,300,350,500,250,250] #heights of the objects >> [para heights, img heights]

col_w = 400 
#every paragraph object has same width
#for image objects, width <= col_w ; but we consider image_width = col_w for computations

#how many are paras, images
num_para,num_im=4,2

#each image is referred by one para
#contains the obj number of corresponding para of images
img_para = [0,4]
# inputs-end --------------------------------------------

#outputs object coordinates as [ [list of x values],[list of y values] ] 

height_dic = {}
q = 0
for o in in_objs_h:
  height_dic[q] = o
  q= q + 1

prob = pulp.LpProblem('killer-layouts', pulp.LpMinimize) 

para_odr = [i for i in range(num_para)]
is_para = []
is_para.append([])
is_para.append([])
for i in range(num_para):
  is_para[0].append(1)
  is_para[1].append(0)
for i in range(num_im):
  is_para[0].append(0)
  is_para[1].append(img_para[i])

print("ispara:",is_para)
print("para odr:",para_odr)
num_all=num_im+num_para

t_canvas_h = 0
for i in range(num_all):
  t_canvas_h = t_canvas_h + height_dic[i]
t_canvas_h = int(t_canvas_h/2)+200  # keping some buffer space
print(t_canvas_h)

obj_permutations=list(permutations([i for i in range(num_all)],num_all))
#print(obj_permutations)

candidates = []
for i in obj_permutations:
  good = 1
  tmp = []
  for idx in range(num_all):
    if(is_para[0][i[idx]]==1):
      tmp.append(i[idx])
  #print("",r,"tmp:",tmp)
  if(tmp == para_odr):
    #print("good tmp",tmp)
    candidates.append(i)
print("candidate layouts:",candidates)

objects = pulp.LpVariable.dicts('objects', candidates, cat='Binary')

#hard constraints--------------------------------------------------------------------------
prob += pulp.lpSum([objects[item] for item in candidates]) == 1

#soft constraints---------------------------------------------------------------------------
def weight(item,mode):
    reward = 0

    #determine the last object of the first column
    t_col_h = 0
    c2first_obj_idx = 0
    for a in item:
      t_col_h = t_col_h + height_dic[a]
      if t_col_h < t_canvas_h:
        c2first_obj_idx=c2first_obj_idx+1
      else:
        break
    #print(c2first_obj_idx)
    col1_h = 0
    col2_h = 0
    col2 = 0
    cols = []
    cols.append([])
    cols.append([])
    y_prev = 0
    x_val = 0
    y = []
    x = []
    idx = 0
    for a in item:
      #print("tp",y_prev,height_dic[a])
      if(idx==c2first_obj_idx): #we hope atlest 1 item fits in to col 1 :)
        col2 = 1
        y.append(0)
        y_prev = height_dic[a]
        x_val = col_w
        x.append(x_val)
      else:
        y.append(y_prev)
        y_prev = y_prev+height_dic[a]
        x.append(x_val)
      if(idx==c2first_obj_idx-1): #if at the last item in col1
        col1_h = y_prev
      elif(idx==num_all-1): #if at the last item in col2
        col2_h = y_prev
      if(col2 == 0):
        cols[0].append(a) #record col1 elements
      else:
        cols[1].append(a) #record col2 elements
      idx=idx+1
    print(item,x,y, col1_h, col2_h)
    
    ############### if mode == 1 return with object coordinates, else continue ###
    if(mode == 1): 
      return x,y

    ##############################################################################
    #soft constraint 3 : it's nice if in a given column two images are not ajacent
    ajacent_img_cost = 0
    for c in cols: #for each column
        i = 0
        for o in c: #for each elements
            if(i > 0 and is_para[0][o]==0 and is_para[0][prev_o]==0):
                ajacent_img_cost = ajacent_img_cost + col_w #this is a sample weight, can change
            prev_o = o
            i = i + 1
    reward =  reward + ajacent_img_cost
    #print("img_cost:",ajacent_img_cost)

    ##############################################################################
    #soft constraint 2 : it's nice if two columns have smaller hight dif
    h_df = 0
    if(col1_h>col2_h):
      h_df = col1_h-col2_h
    else:
      h_df = col2_h-col1_h
    #print("h_df:",h_df)
    reward = reward + h_df

    ##############################################################################
    #soft constraint 1 : it's nice if each image is as close as possible to its para
    #for each image, check the distance to its para
    tot_distance = 0
    img_idx = 0
    for a in is_para[0]:
          if a == 0: # i.e. object is an image
                im_para_no = is_para[1][img_idx] #this img_idx is the object name of image
                #print(im_para_no)
                r=0
                for e in item:
                      if(e == im_para_no): #check if a given object(para) happens to be da matching para
                            para_y = y[r]
                            para_x = x[r]
                            rr = 0
                            for ee in item:
                                  if(ee == img_idx):
                                    img_x = x[rr]
                                    img_y = y[rr]
                                  rr=rr+1
                            break
                      r=r+1
                img_centroid = ( img_x+col_w/2, img_y+height_dic[img_idx]/2 )
                #print("ic",img_centroid)
                para_centroid = (para_x+col_w/2, para_y+height_dic[im_para_no]/2)
                tot_distance = tot_distance +int(np.sqrt( (img_centroid[0]-para_centroid[0])**2+(img_centroid[1]-para_centroid[1])**2 ))
          img_idx = img_idx + 1
    #print("tot dist:", tot_distance)
    reward = reward + tot_distance
    print("weights:",ajacent_img_cost,h_df,tot_distance)
    return reward

prob += pulp.lpSum([objects[item]*weight(item,0) for item in candidates])
#----------------------------------------------------------------------------------------------

# Solve
status = prob.solve()
pulp.LpStatus[status]

for item in candidates:
  if(objects[item].varValue == 1):
    print(item)
    break
[x,y] = weight(item,1)
print("\nkiller-layout:",x,y)