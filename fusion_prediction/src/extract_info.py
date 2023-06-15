#! /usr/bin/env python3

import rospy
import numpy as np
import numpy.linalg as lin
import sys
import os
import cv2
import math
import torch
import pandas as pd
import csv
from time import time

sys.path.append('/home/heven/catkin_ws/src/crat_pred')

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker
import rviz_visualizer as visual
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from collections import deque
from nav_msgs.msg import Odometry
from std_msgs.msg import Time

from model.crat_pred import CratPred

TARGET_ID = 1
OBJECT_NUM = 1

class InfoExtractor():
    def __init__(self):
        
        self.init_time = time()
        self.object_temp_dict = dict()
        self.past_trajs = []
        self.displ_list = []

        self.x_7 = 0
        self.y_7 = 0

        self.odom_list = []
        self.object_list = []

        for i in range(OBJECT_NUM):
            self.object_list.append([])

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
        self.traj_pub = rospy.Publisher('/trajectory', Marker, queue_size=1)

    def odom_callback(self, msg):
        
        self.final_odom_x = msg.pose.pose.position.x

        # print(msg.pose.pose.position.y)

        self.cur_time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9

        print(self.cur_time)
        # print(type(self.cur_time))


    def callback(self, data):
        
        # print(data.header.stamp)
        if (self.frame_cnt == 0):
            for box in data.boxes:

                id = box.label
                
                if id == TARGET_ID or id:
                # if id == TARGET_ID or id == TARGET_ID+1:
                # if id == TARGET_ID or id == TARGET_ID+1 or id == TARGET_ID+2:
                    self.init_time = time()
                    self.prev_time = self.init_time
                    self.frame_cnt = 1

        cur_time = time()

        self.displ_list = []
        
        for i in range(OBJECT_NUM):
            self.displ_list.append([])
        
        origin_list = []

        centers_list = []

        for i in range(OBJECT_NUM):
            centers_list.append([])
        
        centers_list.append([0.0, 0.0])

        if (cur_time - self.prev_time) >= 0.12:
            self.prev_time = cur_time
        
        
        if (((cur_time - self.prev_time) > 0.08 and (cur_time - self.prev_time) < 0.12) or self.frame_cnt == 1):
            
            self.odom_list.append([self.final_odom_x, 0.0, 1.0000])

            # print("dt : ", cur_time - self.prev_time)

            if len(self.odom_list) > 20:
                self.odom_list.pop(0)

            # self.prev_time = cur_time

            for box in data.boxes:

                x = box.pose.position.x
                y = box.pose.position.y
                
                id = box.label

                if id > OBJECT_NUM:
                    id = OBJECT_NUM

                # if id != TARGET_ID:
                # if id != TARGET_ID and id != TARGET_ID+1:
                # if id != TARGET_ID and id != TARGET_ID+1 and id != TARGET_ID+2:
                    # continue

                if id == TARGET_ID:
                    self.x_7 = -x
                    self.y_7 = -y

                centers_list[-(TARGET_ID-id)] = [-x-self.x_7, -y-self.y_7]

                self.object_list[-(TARGET_ID-id)].append([self.odom_list[-1][0] - x, -y, 1.0000])

                for i in range(OBJECT_NUM):
                    if len(self.object_list[i]) > 20:
                        self.object_list[i].pop(0)

                if id == TARGET_ID:
                    origin_list = [self.odom_list[-1][0] - x, -y]
            
            rotation_list = [[1.0, 0.0], [0.0, 1.0]]

            # for i in range(len(self.object_list)):
            #     print("#", i, " length : ", len(self.object_list[i]))


            if len(self.odom_list) == 20 and len(self.object_list[0]) == 20:
                odom_displ = []

                for index in range(len(self.odom_list) - 1):
                    odom_displ.append([(self.odom_list[index+1][0] - self.odom_list[index][0]), (self.odom_list[index+1][1] - self.odom_list[index][1]), 1.0000])

                for object_index in range(len(self.object_list)):
                    for time_index in range(len(self.object_list[object_index]) - 1):
                        self.displ_list[object_index].append([(self.object_list[object_index][time_index+1][0] - self.object_list[object_index][time_index][0]), (self.object_list[object_index][time_index+1][1] - self.object_list[object_index][time_index][1]), 1.0000])

                self.displ_list.insert(OBJECT_NUM,odom_displ)

                # for i in range(len(self.displ_list)):
                #     print("#", i, " length : ", len(self.displ_list[i]))
                

                final_displ = torch.tensor(self.displ_list)
                final_origin = torch.tensor(origin_list)
                final_centers = torch.tensor(centers_list)
                final_rotation = torch.tensor(rotation_list)

                self.final_dict["displ"] = [final_displ]
                self.final_dict["origin"] = [final_origin]
                self.final_dict["centers"] = [final_centers]
                self.final_dict["rotation"] = [final_rotation]
                
                self.result_dict = self.final_dict

                before = time()

                model = CratPred.load_from_checkpoint(checkpoint_path = '/home/heven/catkin_ws/src/crat_pred/final.ckpt')
                model.eval()

                with torch.no_grad():
                    output = model(self.final_dict)
                    after = time()

                    # print("Inference Time : ", after - before)
                    
                    output = [x[0:1].detach().cpu().numpy() for x in output]

                    final_trajectory_list = output[0][0].tolist() # size 6 (probability sorted)

                    final_trajectory = final_trajectory_list[0]

                    # print("Length : ", len(final_trajectory))
                    # print("=================")

                    odom_x = self.odom_list[-1][0]

                    for point in final_trajectory:
                        point[0] = odom_x - point[0]
                        point[1] = -point[1]

                    ids = list(range(29, -1, -1))

                    linelist = visual.points_rviz(name="pointlist", 
                                                    id=ids.pop(),
                                                    points = final_trajectory,
                                                    color_r=255,
                                                    color_g=255,
                                                    color_b=0,
                                                    scale=0.08)
                    
                    self.traj_pub.publish(linelist)

                    # print(type(final_trajectory_list[0][0]))

                    # os.chdir('/home/heven/catkin_ws/src/urp_amlab/csv')

                    # with open('2_prediction_result.csv', 'w') as f:
                    #     w = csv.writer(f)

                    #     for trajectory in final_trajectory_list:
                    #         w.writerow(trajectory)        
    
    def position_2_displ(self, displ_list, traj_list):

        for traj in traj_list:
            
            displ = [] # 얘가 data['displ'][0][0]에 해당

            if len(traj) == 0:
                continue
            
            for index in range(len(traj)-1):
                displ.append([traj[index+1][0] - traj[index][0], traj[index+1][1] - traj[index][1], 1.0000])
            
            displ_list.append(displ)
        
        return displ_list # 얘가 data['displ'][0]에 해당

            






if __name__ == "__main__":

    if not rospy.is_shutdown():
        InfoExtractor()
        rospy.spin()
    
