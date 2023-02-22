import os
from scipy.io import loadmat
import pandas as pd

air_list = os.listdir("images")
annot_list = os.listdir("annotations")
air_list.sort()
annot_list.sort()
coords = []
for i in range(len(air_list)):
    ann = loadmat('annots/' + annot_list[i])
    temp_lst = []
    temp_lst.append(air_list[i])
    temp_lst.append(ann['box_coord'][0][2])
    temp_lst.append(ann['box_coord'][0][0])
    temp_lst.append(ann['box_coord'][0][3])
    temp_lst.append(ann['box_coord'][0][1])
    
    coords.append(temp_lst)

df = pd.DataFrame(coords)
df.to_csv('annotations.csv', index=False, header=None)