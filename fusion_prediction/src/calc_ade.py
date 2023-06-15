import csv
import os
import pandas as pd
from math import *

def get_ade(pred_list, gt_dict, start_index):

    sum = 0
    total_cnt = 0

    for i in range(0, 30):

        if round(float(pred_list[start_index][0]) + 0.1*(i+1), 1) in gt_dict.keys():
            pred_x = float(pred_list[start_index + i][1])
            pred_y = float(pred_list[start_index + i][2])

            gt_x = gt_dict[round(float(pred_list[start_index][0]) + 0.1*(i+1), 1)][0]
            gt_y = gt_dict[round(float(pred_list[start_index][0]) + 0.1*(i+1), 1)][1]

            distance = sqrt(pow(gt_x - pred_x, 2) + pow(gt_y - pred_y, 2))

            # print("distance : ", distance)

            sum += distance
            total_cnt += 1
    
    ade = sum / 30

    return ade

def get_fde(pred_list, gt_dict, start_index):

    if round(float(pred_list[start_index][0]) + 3, 1) in gt_dict.keys():
        pred_x = float(pred_list[start_index + 29][1])
        pred_y = float(pred_list[start_index + 29][2])

        gt_x = gt_dict[round(float(pred_list[start_index][0]) + 3, 1)][0]
        gt_y = gt_dict[round(float(pred_list[start_index][0]) + 3, 1)][1]

        fde = sqrt(pow(gt_x - pred_x, 2) + pow(gt_y - pred_y, 2))

        return fde

    else:
        return 0

os.chdir('/home/heven/catkin_ws/src/urp_amlab/csv/result')

f = open('2_result.csv', 'r', encoding='utf-8-sig')
rea = csv.reader(f, delimiter=',')
pred_result = list(rea)


gt_f = open("2_gt_result.csv", 'r', encoding='utf-8-sig')
gt_rea = csv.reader(gt_f, delimiter=',')

gt_result = dict()

for row in gt_rea:
    gt_result[float(row[0])] = [float(row[1]), float(row[2])]

cnt = 0

ade_list = []
fde_list = []

for pred in pred_result:
    
    if (cnt % 30) == 0:
        ade = get_ade(pred_result, gt_result, cnt)
        fde = get_fde(pred_result, gt_result, cnt)
        
        if ade != 0:
            ade_list.append(ade)
        
        if fde != 0:
            fde_list.append(fde)

    cnt += 1

print(min(ade_list))
print(min(fde_list))