#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:42:58 2022

@author: yejohnny
"""


#plot movement sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from random import sample
import matplotlib.patches as patches
import random
import matplotlib.patches as patches

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-MOTP_MODP"
ouput_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure7_Movement_sequence_prediction\predicted_movement_sequences'


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if file_name.startswith('rec-'):
            USN = file_name.split('-')[1]
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Movement_Label_path = get_path(InputData_path_dir,'Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')


movement_order = ['running','trotting','left_turning','right_turning','walking','stepping','sniffing',
                  'rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

movement_color_dict = {'running':'#FF3030',
                       'trotting':'#E15E8A',                       
                       'left_turning':'#F6BBC6', 
                       'right_turning':'#F8C8BA',
                       'walking':'#EB6148',
                       'stepping':'#C6823F',  
                       'sniffing':'#2E8BBE',
                       'rising':'#84CDD9',
                       'hunching':'#D4DF75',
                       'rearing':'#88AF26',
                       'climbing':'#2E7939',                           
                       'jumping':'#24B395',
                       'grooming':'#973C8D',
                       'scratching':'#EADA33',
                       'pausing':'#B0BEC5',}


movement_frequency_each_mice = []


event_list = []
mouse_id_list = []
male_num = 0

for index in random.sample(list(Feature_space_path.keys()), k=5):

    mouse_id_list.append('new_rondom{}'.format(index))
    FeA_file = pd.read_csv(Feature_space_path[index])
    
    start = 0
    end = 0

    segBoundary_dict = {}
    
    for index2 in range(FeA_file.shape[0]):
        #movement_label = return_AnnoMovement(int(FeA_file.loc[index,'movement_label']))
        movement_label = FeA_file.loc[index2,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index2,'segBoundary_end']
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    event_list.append(segBoundary_dict)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30,5),constrained_layout=True,dpi=300)

num = 0
for i in range(len(event_list)):
    mouse_event = event_list[i]
    mouse_id = mouse_id_list[i]
    for key in mouse_event.keys():
        ax.add_patch(
        patches.Rectangle(
            (0, num-0.02),   # (x,y)
            30*60*60,          # width
            0.54,          # height
            #alpha=0.3,ec="#FF7F50", fc="#FFBF00",lw=5)
            alpha=1,ec="black", fc='white',lw=3),

            )
        color = movement_color_dict[key]
        ax.broken_barh(mouse_event[key],(num,0.5),facecolors=color,zorder=2)
        #plt.text(-9000,num+0.2,mouse_id,fontsize=20)

    num +=0.6


plt.title('predicted movement sequences ({})'.format(InputData_path_dir.split('/')[-1].split('-')[-1]),fontdict={'size':30},horizontalalignment='center')
plt.axis("off")
plt.savefig('{}/{}_movement_sequences.png'.format(ouput_dir,InputData_path_dir.split('/')[-1].split('-')[-1]),dpi=300)