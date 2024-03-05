# Human Pose Estimation using Zed2i Camera

## Overview
This repository contains code for performing human pose estimation using the Zed2i camera fused with depth. The project leverages computer vision techniques to analyze human poses from the camera feed.
The goal of this project is to get human pose estimation and localize it in 3D space. Using this information, analyze human movements in different environmental conditions.


## Features
-**Real-time Pose Estimation:** Utilize the Zed2i camera to perform real-time human pose estimation.
-**Configuration:** Easily configure and customize the parameters of the camera for pose estimation according to your requirements.
-**Applications:** Analyse movements of humans in different conditions of the room. Eventually, make the room conditions such that the person is relaxed.

## Prerequisites
-Zed2i Camera
-Zed SDK and its requirements.
-Python
-Nvidia-GPU

## Installation
1. Install [Zed SDK][https://github.com/stereolabs/zed-sdk]
2. Clone the [Repository][git clone https://github.com/venk221/human_pose_estimation-using-Zed2i-camera.git]
3. To perform real-time pose estimation, run the file using: python "mixture.py" --output_svo_file "path\to\save\file.svo"
4. To export the svo file into avi file run the following script from the Zed SDK: python svo_export.py --mode "0/1/2/3" --input_svo_file path\to\svo\file.svo --output_avi_file path\to\file.avi
   
## Analysis:
Analysis is still ongoing. 



