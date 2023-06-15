#! /usr/bin/env python3

import rospy
import torch
import sys
from pathlib import Path
import os

from time import time

from jsk_recognition_msgs.msg import BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from collections import deque

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

import rviz_visualizer as visual


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0] / "crat-pred"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

from model.crat_pred import CratPred

class Predictor():
    def __init__(self):
        
        self.id_list = []

        self.global_trajectory = dict()
        self.global_trajectory[0] = []

        self.init_flag = 0
        self.ref_odom_x = 0
        self.centers_list = dict()
        self.past_trajectory = dict()

        rospy.init_node('prediction_node', anonymous=False)
        self.sensor_direction = rospy.get_param("~sensor_direction")
        self.crat_pred_path = rospy.get_param("~crat_pred_path")

        
        rospy.Subscriber('fusion/tracks', BoundingBoxArray, self.tracking_callback)
        rospy.Subscriber('/Odometry', Odometry, self.odom_callback)
        self.traj_pub = rospy.Publisher('/trajectory', MarkerArray, queue_size=1)

    
    def odom_callback(self, msg):

        if self.init_flag == 0:
            self.ref_odom_x = msg.pose.pose.position.x
            self.init_flag = 1

        self.odom_x = msg.pose.pose.position.x - self.ref_odom_x
        self.odom_y = msg.pose.pose.position.y

        self.cur_time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9

    
    def tracking_callback(self, data):
        
        self.object_boxes = data.boxes
        self.prediction()
    

    def prediction(self):
        
        if (self.cur_time % 0.1) > 0.09 or (self.cur_time % 0.1) < 0.01:
            
            tmp_displ_list = []
            
            rotation_list = [[1.0, 0.0], [0.0, 1.0]]

            possible_idx_list = []
            id_order = []

            # ego vehicle Odometry append
            self.global_trajectory[0].append([self.odom_x, 0.0, 1.0000])
            

            # when length of ego vehicle Odometry > 20, then pop the oldest data
            if len(self.global_trajectory[0]) > 20:
                self.global_trajectory[0].pop(0)

                odom_displ = self.position_2_displ(self.global_trajectory[0])

                tmp_displ_list.append(odom_displ)

                id_order.append(0)

            # check if the object data is available to be preprocessed (if the data is 20)
            for box in self.object_boxes:
                
                x = box.pose.position.x
                y = box.pose.position.y

                # empty list assignment
                if box.label not in self.global_trajectory.keys():
                    self.global_trajectory[box.label] = []
                
                # Last odometry value
                last_odom = self.global_trajectory[0][-1][0]

                # Object global coordinate append
                if self.sensor_direction == "rear":
                    self.global_trajectory[box.label].append([last_odom - x, -y, 1.0000])
                if self.sensor_direction == "front":
                    self.global_trajectory[box.label].append([last_odom + x, y, 1.0000])

                # check if length of the global coordinate of the object is over than 20
                if len(self.global_trajectory[box.label]) > 20:
                    self.global_trajectory[box.label].pop(0)
                    possible_idx_list.append(box.label)
            
            # Convert global coordinates into displacements
            for idx in possible_idx_list:

                obj_displ = self.position_2_displ(self.global_trajectory[idx])

                tmp_displ_list.append(obj_displ)

                id_order.append(idx)
            
            # Queue for object id
            id_queue = deque(id_order)

            # Queue for object displacement
            displ_queue = deque(tmp_displ_list)

            # MarkerArray for the trajectory visualization
            multi_agent_trajectory = MarkerArray()

            # Before inference
            before = time()

            # ids for the visualization (ids = [29, 28, ..., 0])
            ids = list(range(29*len(id_order),-1, -1))

            ####################################################
            ###### Multi-agent Trajectory Prediction loop ######
            ####################################################
            for _ in range(len(id_order)):
                
                # dict for the CRAT-Pred input
                final_dict = dict()

                centers_list = []
                
                # target_id -> first element of the id_queue is assigned
                target_id = list(id_queue)[0]
                
                # target vehicle global x, y coordinate
                target_x = self.global_trajectory[target_id][-1][0]
                target_y = self.global_trajectory[target_id][-1][1]

                # target vehicle global x, y coordinate list
                origin_list = [target_x, target_y]

                # for loop for calculating each local coordinate about the target vehicle
                for id in list(id_queue):
                    
                    cur_x = self.global_trajectory[id][-1][0]
                    cur_y = self.global_trajectory[id][-1][1]
                    
                    # local x, y coordinate of all agents when t = 0 (current)
                    centers_list.append([cur_x - target_x, cur_y - target_y])
                
                # Convert displacement Queue into list
                displ_list = list(displ_queue)

                # Convert all list into tensor
                final_displ = torch.tensor(displ_list)
                final_origin = torch.tensor(origin_list)
                final_centers = torch.tensor(centers_list)
                final_rotation = torch.tensor(rotation_list)

                # Assign all tensors to the value of the input dict
                final_dict["displ"] = [final_displ]
                final_dict["origin"] = [final_origin]
                final_dict["centers"] = [final_centers]
                final_dict["rotation"] = [final_rotation]

                # CRAT-Pred
                model = CratPred.load_from_checkpoint(checkpoint_path = self.crat_pred_path)
                model.eval()

                with torch.no_grad():
                    output = model(final_dict)

                    output = [x[0:1].detach().cpu().numpy() for x in output]

                    final_trajectory_list = output[0][0].tolist() # size 6 (probability sorted)

                    final_trajectory = final_trajectory_list[0]

                    odom_x = self.odom_x

                    # Convert the output global coordinate into local coordinate about ego vehicle
                    for point in final_trajectory:
                        if self.sensor_direction == "rear":
                            point[0] = odom_x - point[0]
                            point[1] = -point[1]
                        
                        if self.sensor_direction == "front":
                            point[0] = point[0] - odom_x
                            point[1] = point[1]
                    
                    linelist = visual.points_rviz(name="pointlist", 
                                                    id=ids.pop(),
                                                    points = final_trajectory,
                                                    color_r=255,
                                                    color_g=255,
                                                    color_b=0,
                                                    scale=0.2)
                    
                    multi_agent_trajectory.markers.append(linelist)
                
                id_queue.rotate(-1)
                displ_queue.rotate(-1)
            
            # After inference
            after = time()

            # Print inference time
            print("Multi-agent Trajectory Inference time : ", after - before)

            self.traj_pub.publish(multi_agent_trajectory)
            
    
    def position_2_displ(self, global_traj):

        displ = []
        for index in range(len(global_traj)-1):
            displ.append([global_traj[index+1][0] - global_traj[index][0], global_traj[index+1][1] - global_traj[index][1], 1.0000])
            
        return displ

if __name__ == "__main__":
    
    if not rospy.is_shutdown():
        Predictor()
        rospy.spin()