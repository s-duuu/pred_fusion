#! /usr/bin/env python3

import rospy
import os
import numpy as np
import numpy.linalg as lin
import sys
import cv2
import math
import torch
import pandas as pd
import csv
from time import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from collections import deque
from nav_msgs.msg import Odometry
from trajectory_prediction.msg import gt_object

class DataGenerator():
    def __init__(self):
        
        self.init_time = time()
        self.object_temp_dict = dict()
        self.past_trajs = []
        self.displ_list = []

        self.odom_list = []
        self.object_list = [[], []]

        self.time_list = []
        
        self.final_dict = dict()
        self.result_dict = dict()
        self.frame_cnt = 0
        self.prev_time = 0

        self.final_flag = 0

        # CRAT-Pred input 용 dictionary
        for i in range(1, 10):
            self.object_temp_dict[i] = []
            self.past_trajs.append([])
        
        rospy.init_node('extractor_node', anonymous=False)
        rospy.Subscriber('fusion/tracks', BoundingBoxArray, self.callback)
        rospy.Subscriber('/Odometry', Odometry, self.odom_callback)

    def odom_callback(self, msg):

        self.final_odom_x = msg.pose.pose.position.x

    def callback(self, data):

        if (self.frame_cnt == 0):
            self.init_time = time()
            self.prev_time = self.init_time
            self.frame_cnt = 1

        cur_time = time()

        
        displ_list = []

        for i in range(3):
            displ_list.append([])
        
        origin_list = []
        centers_list = [[0.0, 0.0],[],[]]

        # tmp_object_list = []
        
        if (((cur_time - self.prev_time) > 0.08 and (cur_time - self.prev_time) < 0.12) or self.frame_cnt == 0):
            
            self.odom_list.append([self.final_odom_x, 0.0, 1.0000])

            self.prev_time = cur_time

            for box in data.boxes:
                x = box.pose.position.x
                y = box.pose.position.y
                
                id = box.label

                print(id)

                centers_list[id] = [-x, -y]

                self.object_list[id-1].append([self.odom_list[-1][0] - x, -y, 1.0000])

                if id == 1:
                    origin_list = [self.odom_list[-1][0] - x, -y]
            
            rotation_list = [[1.0, 0.0], [0.0, 1.0]]

            displ_list = [[], []]

            if len(self.odom_list) == 20 and len(self.object_list[1]) == 20:
                odom_displ = []

                for index in range(len(self.odom_list) - 1):
                    odom_displ.append([(self.odom_list[index+1][0] - self.odom_list[index][0]), (self.odom_list[index+1][1] - self.odom_list[index][1]), 1.0000])

                for object_index in range(len(self.object_list)):
                    for time_index in range(len(self.object_list[object_index]) - 1):
                        displ_list[object_index].append([(self.object_list[object_index][time_index+1][0] - self.object_list[object_index][time_index][0]), (self.object_list[object_index][time_index+1][1] - self.object_list[object_index][time_index][1]), 1.0000])

                displ_list.insert(0, odom_displ)

                final_displ = torch.tensor(displ_list)
                final_origin = torch.tensor(origin_list)
                final_centers = torch.tensor(centers_list)
                final_rotation = torch.tensor(rotation_list)

                self.final_dict["displ"] = [final_displ]
                self.final_dict["origin"] = [final_origin]
                self.final_dict["centers"] = [final_centers]
                self.final_dict["rotation"] = [final_rotation]
                
                self.result_dict = self.final_dict

            
            # if len(self.odom_list) == 21:
            #     print("!!!!!!Last x : ", self.odom_list[20])


if __name__ == "__main__":

    if not rospy.is_shutdown():
        data_gen = DataGenerator()
        rospy.spin()
    
    os.chdir('/home/heven/catkin_ws/src/urp_amlab/csv')

    with open('dataset_test.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(data_gen.result_dict.keys())
        w.writerow(data_gen.result_dict.values())
    