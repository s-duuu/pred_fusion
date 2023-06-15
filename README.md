# ROS Sensor Fusion based Multi-Object Trajectory Prediction

This repository deals with a perception system of Autonomous Driving techniques. In particular, we focused on the object detection, tracking, sensor fusion, and trajectory prediction. We used YOLOv5, PointPillars for the object detection of Camera and LiDAR sensor, respectively. Overall pipeline is as following.

<p align="center">
    <img src="https://github.com/s-duuu/pred_fusion/assets/96241852/4fac99cd-24b7-418f-ab5a-65c4ec965663" align="center" width="80%">
</p>

## System Prediction Result

<p align="center">
    <img src="https://github.com/s-duuu/pred_fusion/assets/96241852/82540016-3868-4f8c-be07-0eb9b963657f" align="center" width="40%">
    <img src="https://github.com/s-duuu/pred_fusion/assets/96241852/702726ad-bd41-4beb-9c14-1577e0575d26" align="center" width="40%">
</p>

Through ROS Rviz, the prediction output is as the videos above.

## Prerequisite
1. Tested in Ubuntu 20.04 (ROS Noetic) & NVIDIA GeForce RTX 3070
2. Other necessary library is in the `requirements.txt`

## Preparation
### 0. Clone this repository and move directory
Clone this repository and move your current directory to here.

    cd path_to_your_ws
    git clone https://github.com/s-duuu/pred_fusion.git
    cd pred_fusion

### 1. Install requirements
Install modules in `requirements.txt`.

    pip install -r requirements.txt
    
### 2. Clone PointPillars
Clone the [official repository of PointPillars](https://github.com/zhulf0804/PointPillars).

    git clone https://github.com/zhulf0804/PointPillars.git

### 3. Clone OpenPCDet
Clone the [official repository of OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    git clone https://github.com/open-mmlab/OpenPCDet.git

### 4. Clone CRAT-Pred
Clone the [official repository of CRAT-Pred](https://github.com/schmidt-ju/crat-pred).

    git clone https://github.com/schmidt-ju/crat-pred.git

### 5. Build package
Build the package in the your workspace.

    cd path_to_your_ws
    catkin_make (or catkin build)
    source ./devel/setup.bash

## Execute Launch file & Test
Execute launch file which includes all ROS nodes necessary for the system.

    roslaunch fusion_prediction integrated.launch

You can test our system by [ROS bagfile](https://drive.google.com/file/d/1xxUuHh4EdGnaSU-Z3uUCBFW6_bFpsDiu/view?usp=sharing). Download the file and play it in another terminal. Rviz will display the result of the system.

    cd path_to_bagfile
    rosbag play test.bag

    
## Detection Models
### 1. YOLOv5
We trained YOLOv5s model, which is located in `pred_fusion/fusion_prediction/yolo.pt`. Since the model was trained with image data extracted from CarMaker simulator, if you need the YOLOv5 model for the real vehicles, it would be better to change the YOLO model. You can train a new model from [yolov5 official github](https://github.com/ultralytics/yolov5).

### 2. PointPillars
We also trained PointPillars model, which is located in `pred_fusion/fusion_prediction/pillars.pth`. This model was trained with Kitti dataset, thus you don't need to change the model.

## Sensor Fusion
Sensor fusion algorithm is based on Late Fusion algorithm. Algorithm in this repository is based on the bounding box projection. Each 3D bounding box predicted from the PointPillars model is projected onto the image plane. Then, the algorithm determines whether the 2 bounding boxes are for the same object based on IOU.

<p>
  <img src="https://github.com/s-duuu/pred_fusion/assets/96241852/667dfe8b-9bc8-4b1c-a8f4-a762cac71b95" alt="" width="400">
</p>

## Object Tracking
Object tracking algorithm is based on the [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort). The algorithm tracks each BEV (Bird's Eye View) Bounding Box. Tracking is based on Kalman Filter, Matching is based on IOU, and Assignment is based on Hungarian algorithm.

<p align="center">
  <img src="https://github.com/s-duuu/pred_fusion/assets/96241852/50035308-1660-4c32-8b5e-f411d3303eb8" align="center" width="49%">
  <img src="https://github.com/s-duuu/pred_fusion/assets/96241852/e3baeefe-6f12-4a1a-837d-5d4bf0bb9d8c" align="center" width="49%">
</p>

## Trajectory Prediction
Trajectory is predicted from the [CRAT-Pred model](https://github.com/schmidt-ju/crat-pred). This model was trained with Argoverse dataset, thus you don't need to change the model. The model is located in `pred_fusion/fusion_prediction/crat.ckpt`.

## Contributor
Kim SeongJu, School of Mechanical Engineering, Sungkyunkwan University, South Korea

e-mail: sungju29@g.skku.edu
