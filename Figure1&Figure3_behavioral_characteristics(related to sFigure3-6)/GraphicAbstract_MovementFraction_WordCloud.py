# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:54:36 2024

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud,get_single_color_func
import matplotlib.pyplot as plt
from matplotlib import colors


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Figure1&Figure3_behavioral_characteristics(related to sFigure3-6)\MovementFraction_WordCloud'
if not os.path.exists(output_dir):                                    
    os.mkdir(output_dir)                                             


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'Movement_Labels.csv')       
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')


skip_file_list = [1,3,28,29,110,122] 

animal_info_csv = r'F:\spontaneous_behavior\GitHub\The-hierarchical-behavioral-analysis-framework\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]


Morning_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentCondition']=='Night') & (animal_info['LightingCondition']=='Light-off')]

Stress_info = animal_info[(animal_info['ExperimentCondition']=='Stress')]

movement_order2 = ['running','trotting','left_turning','right_turning','walking','stepping',
                  'sniffing','rising','hunching','rearing','climbing','jumping','grooming','scratching','pausing',]

movement_order = ['Running','Trotting','Left turning','Right turning','Walking','Stepping','Sniffing',
                 'Rising','Hunching','Rearing','Climbing','Jumping','Grooming','Scratching','Pausing',]

big_cluster_dict4 = {'locomotion':['running','trotting','walking','left_turning','right_turning','stepping'],
                     'exploration':['climbing','rearing','hunching','rising','sniffing','jumping'],
                     'maintenance':['grooming','scratching'],
                     'nap':['pausing'],
                     } 

exploration_move=['climbing','rearing','hunching','rising','sniffing','jumping']

movement_color_dict = {'running':'#FF3030',
                       'trotting':'#E15E8A',                       
                       'left_turning':'#F6BBC6', 
                       'right_turning':'#F8C8BA',
                       'walking':'#EB6148',
                       'stepping':'#C6823F',  
                       'sniffing':'#2E8BBE',
                       'rising':'#84CDD9',    #'#FFEA00'  ####FFEE58
                       'hunching':'#D4DF75',
                       'rearing':'#88AF26',
                       'climbing':'#2E7939',                           
                       'jumping':'#24B395',                                              
                       'grooming':'#973C8D',
                       'scratching':'#EADA33',
                       'pausing':'#B0BEC5',}




def extract_segment(df):                                                        ## Only use movement fractions as features    
    df_mov_fra = pd.DataFrame()
    label_count = df.value_counts('revised_movement_label')

    num = 0
    for mv in movement_order:
        if mv in label_count.index:
            count = label_count[mv]
        else:
            count = 0
        df_mov_fra.loc[num,'movement_label'] = mv
        df_mov_fra.loc[num,'movement_fraction'] = count
        num += 1
    df_mov_fra['movement_fraction'] = (df_mov_fra['movement_fraction'] /df_mov_fra['movement_fraction'].sum())*100
    return(df_mov_fra)


dataset = Stress_info
dataset_name = 'Stress'


all_matrix = []
for index in dataset.index:
    video_index = dataset.loc[index,'video_index']
    gender =  dataset.loc[index,'gender']
    ExperimentCondition =  dataset.loc[index,'ExperimentCondition']
    LightingCondition = dataset.loc[index,'LightingCondition']
    MoV_file = pd.read_csv(Movement_Label_path[video_index])
    MoV_file.reset_index(drop=True,inplace=True)
    
    
    all_matrix.append(MoV_file)


All_df = pd.concat(all_matrix)
All_df.replace({'revised_movement_label': 
                {'running':'Running', 
                 'trotting':'Trotting',
                 'left_turning':'Left turning',
                 'right_turning':'Right turning',
                 'walking':'Walking',
                 'stepping':'Stepping',
                 'sniffing':'Sniffing',
                 'rising':'Rising',
                 'hunching':'Hunching',
                 'rearing':'Rearing',
                 'climbing':'Climbing',
                 'jumping':'Jumping',
                 'grooming':'Grooming',
                 'scratching':'Scratching',
                 'pausing':'Pausing',
                 }},inplace=True)
MoV_frac = extract_segment(All_df)


class SimpleGroupedColorFunc(object):
    def __init__(self, color_to_words, default_color):
        # 特定词颜色
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}
        # 默认词颜色
        self.default_color = default_color
 
    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)




color_to_words  = {'#FF3030':['Running'],
                    '#E15E8A':['Trotting'],                       
                    '#F6BBC6':['Left turning'], 
                    '#F8C8BA':['Right turning'],
                    '#EB6148':['Walking'],
                    '#C6823F':['Stepping'],  
                    '#2E8BBE':['Sniffing'],
                    '#84CDD9':['Rising'],    #'#FFEA00'  ####FFEE58
                    '#D4DF75':['Hunching'],
                    '#88AF26':['Rearing'],
                    '#2E7939':['Climbing'],                           
                    '#24B395':['Jumping'],                                              
                    '#973C8D':['Grooming'],
                    '#EADA33':['Scratching'],
                    '#B0BEC5':['Pausing'],}


default_color = 'grey'

# set movement color
grouped_color = SimpleGroupedColorFunc(color_to_words, default_color)
 

word_frequencies = {}
for i in MoV_frac.index:
    word_frequencies.setdefault(MoV_frac.loc[i,'movement_label'],MoV_frac.loc[i,'movement_fraction'])


# plot word cloud based on movement fration 
wordcloud = WordCloud(background_color="white",random_state=None,width=1200,height=1000,mode='RGBA',).generate_from_frequencies(word_frequencies)
wordcloud.recolor(color_func=grouped_color)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.savefig('{}/{}_movement_fraction_word_cloud.png'.format(output_dir,dataset_name), dpi=300, transparent=True)

plt.show()
