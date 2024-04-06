import sys
import numpy as np
import os  # Add this import for accessing file paths
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QStackedWidget, QSlider, QLabel, QHBoxLayout
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtCore import QUrl  # Add this import for QUrl
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt # Text
from PyQt5.QtGui import QPixmap  # Import QPixmap for handling images
import serial
import time
import numpy as np
import cv2
from pylsl import StreamInlet, resolve_stream
import pandas as pd
import threading
import datetime
import subprocess
import boto3
import os

import cv2
import sys
import pyzed.sl as sl
import numpy as np
import argparse
from signal import signal, SIGINT
import os
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

from pathlib import Path
import enum


'''
Wall 1 motor 1- top of left wall
Wall 1 motor 2- middle of left wall
Wall 1 motor 3- bottom of left wall
Wall 1 motor 4 - left part of celing
Wall 2 motor 1 - top of close wall
Wall 2 motor 2 - middle of close wall
Wall 2 motor 3 - bottom of close wall
Wall 3 motor 4 - right part of the celing
Roof motor 2 & 4 dont work
'''


# Replace the serial port names with the appropriate ones for your Arduino.
# The first two arduino gets to the roof control
# The roof motors are number 4
serial_ports = ['COM3' ,'COM4' ,'COM5' ,'COM6']
# number of motors
motor_num = 16
# sleep period
sleep_prd = 1
# park period
park_prd = 14
# Motor modes
STOP     = 0
FORWARD  = 1
BACKWARD = 2

# ----- Create a list of serial objects
serial_objects = [serial.Serial(port, 9600) for port in serial_ports]

#Walls al the way in, 0 is in 15 is out
prv_setup = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
cur_stat = prv_setup

class MotorControlGUI(QMainWindow):
    #matches motors with walls, 1 through 4 each corresponding with a wall
    motor_serial = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    #Assigns each motor a position on the wall.
    motor_pin    = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])

    def __init__(self):
        super().__init__()
        self.park_motors()#Resets motors to be all the way in
        self.setWindowTitle("Motor Control GUI")
        self.setGeometry(100, 100, 400, 300)

        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        #the different predetermined setups
        self.setup_A = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])#All the way in
        self.setup_B = np.array([[15, 15, 15, 15], [15, 15, 15, 15], [15, 15, 15, 15], [15, 15, 15, 15]])#All the way out
        self.setup_C = np.array([[7, 7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7]])#Halfway
        self.setup_D = np.array([[5, 5, 7, 15], [15, 0, 13, 4], [4, 0, 15, 3], [15, 6, 0, 10]])
        self.setup_H = np.array([[0, 0, 12, 7], [13, 2, 5, 5], [0, 2, 11, 7], [8, 8, 8, 7]])

        # Inside your MotorControlGUI class constructor
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor("#F2E0C9"))  # Set background color
        self.setPalette(palette)


        self.page_1 = QWidget()
        layout_1 = QVBoxLayout()

        # Add title page for the GUI
        title_label = QLabel("PEACE\nPersonal Emotional Augmented Controlled Environment")
        title_label.setStyleSheet("color: #393640; font-size: 17px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)  # Center-align the text

        # Add Image
        image_label = QLabel()
        pixmap = QPixmap(r"D:\GUI_data\PEACE.png")
        image_height_inches = 2.5  # Desired height in inches
        image_height_pixels = int(image_height_inches * pixmap.logicalDpiY())  # Convert inches to pixels
        pixmap = pixmap.scaledToHeight(image_height_pixels, Qt.SmoothTransformation)  # Scale pixmap to desired height
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        layout_1.addWidget(title_label)
        layout_1.addWidget(image_label)  #

        #buttons for each of the walls
        wall_button_1 = QPushButton("Wall 1")
        wall_button_1.clicked.connect(lambda: self.show_wall_page(1))
        wall_button_2 = QPushButton("Wall 2")
        wall_button_2.clicked.connect(lambda: self.show_wall_page(2))
        wall_button_3 = QPushButton("Wall 3")
        wall_button_3.clicked.connect(lambda: self.show_wall_page(3))
        wall_button_4 = QPushButton("Wall 4")
        wall_button_4.clicked.connect(lambda: self.show_wall_page(4))
        # wall_button_5 = QPushButton("Camera")
        # wall_button_5.clicked.connect(lambda: self.show_wall_page(5))
        layout_1.addWidget(wall_button_1)
        layout_1.addWidget(wall_button_2)
        layout_1.addWidget(wall_button_3)
        layout_1.addWidget(wall_button_4)
        # layout_1.addWidget(wall_button_5)
        self.page_1.setLayout(layout_1)
        self.central_widget.addWidget(self.page_1)


        self.page_1.setLayout(layout_1)
        self.central_widget.addWidget(self.page_1)

        # Add a button to navigate to the Roof page
        roof_button = QPushButton("Roof")
        roof_button.clicked.connect(self.show_roof_page)
        layout_1.addWidget(roof_button)

        #Participant Button - opens panas and consent form
        participant_button = QPushButton("Participant Page")
        participant_button.clicked.connect(self.participant_loop)
        layout_1.addWidget(participant_button)

        #Data Button
        data_button = QPushButton("Data Page")
        data_button.clicked.connect(self.show_data)
        layout_1.addWidget(data_button)

        self.page_1.setLayout(layout_1)
        self.central_widget.addWidget(self.page_1)


        #Create the Roof Page
        self.roof_page = self.create_roof_page()
        self.central_widget.addWidget(self.roof_page)

        #Creates buttons for the presets
        setup_button_A = QPushButton("Apply Setup A (Walls In)")
        setup_button_B = QPushButton("Apply Setup B (Walls out)")
        setup_button_C = QPushButton("Apply Setup C (Halfway)")
        setup_button_D = QPushButton("Apply Setup D")
        setup_button_H = QPushButton("Apply Setup H")

        setup_button_A.clicked.connect(lambda: self.apply_setup(self.setup_A))
        setup_button_B.clicked.connect(lambda: self.apply_setup(self.setup_B))
        setup_button_C.clicked.connect(lambda: self.apply_setup(self.setup_C))
        setup_button_D.clicked.connect(lambda: self.apply_setup(self.setup_D))
        setup_button_H.clicked.connect(lambda: self.apply_setup(self.setup_H))

        layout_1.addWidget(setup_button_A)
        layout_1.addWidget(setup_button_B)
        layout_1.addWidget(setup_button_C)
        layout_1.addWidget(setup_button_D)
        layout_1.addWidget(setup_button_H)


        # Initialize the media player
        self.media_player = QMediaPlayer()

        #creates the page for the sounds
        self.music_page = QWidget()
        layout_music = QVBoxLayout()

        #Music Button
        music_button = QPushButton("Music Page")
        music_button.clicked.connect(self.show_music_page)
        layout_1.addWidget(music_button)

        #Reset Button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(lambda: self.reset_sliders())
        layout_1.addWidget(reset_button)

        #Send Motor Button
        send_button = QPushButton("Send Motor Values")
        send_button.clicked.connect(self.send_motor_control)
        layout_1.addWidget(send_button)

        #creates wall pages for each wall
        self.wall_pages = []
        for wall_num in range(1, 5):
            wall_page = self.create_wall_page(wall_num)
            self.wall_pages.append(wall_page)
            self.central_widget.addWidget(wall_page)

        self.motor_values = [[0, 0, 0, 0] for _ in range(4)]  # Initial values for each wall's motors
        self.global_motor_values = np.array(self.motor_values)  # Global numpy array



        #Music Page
        ocean_button = QPushButton("Ocean")
        ocean_button.clicked.connect(lambda: self.play_music(r"D:\GUI_data\Music-20240401T214135Z-001\Music\ocean rolling waves.mp3"))
        rain_button = QPushButton("Rain")
        rain_button.clicked.connect(lambda: self.play_music(r"D:\GUI_data\Music-20240401T214135Z-001\Music\light rain.mp3"))
        birds_button = QPushButton("Birds")
        birds_button.clicked.connect(lambda: self.play_music(r"D:\GUI_data\Music-20240401T214135Z-001\Music\morning birds chirp.mp3"))
        waterfall_button = QPushButton("Waterfall")
        waterfall_button.clicked.connect(lambda: self.play_music(r"D:\GUI_data\Music-20240401T214135Z-001\Music\waterfall.mp3"))
        river_button = QPushButton("River")
        river_button.clicked.connect(lambda: self.play_music(r"D:\GUI_data\Music-20240401T214135Z-001\Music\running river.mp3"))

        layout_music.addWidget(ocean_button)
        layout_music.addWidget(rain_button)
        layout_music.addWidget(birds_button)
        layout_music.addWidget(waterfall_button)
        layout_music.addWidget(river_button)

        stop_button = QPushButton("Stop Music")
        stop_button.clicked.connect(self.stop_music)
        layout_music.addWidget(stop_button)

        back_button_music = QPushButton("Back")
        back_button_music.clicked.connect(self.show_home_page)
        layout_music.addWidget(back_button_music)

        buttons = [wall_button_1, wall_button_2, wall_button_3, wall_button_4,
        setup_button_A, setup_button_B, setup_button_C, setup_button_D, setup_button_H,
        music_button, reset_button, wall_button_2, rain_button,
        back_button_music, send_button]  # Add the new button here
        # button_palette = QPalette()

        # Add the new button here
        """buttons = [wall_button_1, wall_button_2, wall_button_3, wall_button_4,
        setup_button_A, setup_button_B, setup_button_C, setup_button_D, setup_button_H,
        music_button, reset_button, exit_button, soft_noise_button, rain_button, wind_button,
        back_button_music, send_button]  # Add the new button here
        button_palette = QPalette()

        button_stylesheet = "background-color: #BFA995; color: #393640;"
        for button in buttons:
            button.setStyleSheet(button_stylesheet)

        # Inside your MotorControlGUI class constructor, after creating buttons
        text_palette = QPalette()
        text_palette.setColor(QPalette.ButtonText, QColor("#393640"))  # Set text color
        for button in buttons:
            button.setPalette(text_palette)"""

        #Data Page
        self.dataPage = QWidget()
        dataLayout = QVBoxLayout()

        data_camera_button = QPushButton("Record Camera")
        data_camera_button.clicked.connect(self.record_Camera_Loop)
        dataLayout.addWidget(data_camera_button)

        converter = QPushButton("Convert into .AVI")
        converter.clicked.connect(self.convert_Camera_Loop)
        dataLayout.addWidget(converter)

        data_EEG_button = QPushButton("Record EEG")
        data_EEG_button.clicked.connect(self.record_EEG_Loop)
        dataLayout.addWidget(data_EEG_button)

        data_watch_button = QPushButton("Download Watch Data")
        data_watch_button.clicked.connect(self.download_csv_from_s3)
        dataLayout.addWidget(data_watch_button)

        data_back_button = QPushButton("Back")
        data_back_button.clicked.connect(self.show_home_page)
        dataLayout.addWidget(data_back_button)

        self.dataPage.setLayout(dataLayout)
        self.central_widget.addWidget(self.dataPage)


        self.music_page.setLayout(layout_music)
        self.central_widget.addWidget(self.music_page)  # Add the music page to the stacked widget
        #self.central_widget.setCurrentWidget(self.music_page)  # Set initial current widget to music page


    def download_csv_from_s3(self): # add argument that will allow
        #temp.txt is where the id for the participant is stored
        with open(r"D:\GUI_data\temp.txt", 'r') as f:#temp.txt is where the id for the participant is stored
            ID = f.read()

        # Replace these values with your actual credentials and file details
        access_key = 'AKIAWWZYTIF5UYUDU44W'
        secret_key = 'UNDhqvPU8az+PrrQxZS7v6CsTmZ6ZS7YRyaEXPFR'
        bucket_name = 'empatica-us-east-1-prod-data'
        prefix = 'v2/809/'  # Replace with the object (file) key in the bucket
        local_directory = rf"D:\New Data PEACE Experiment\{ID}" # Local directory where files will be saved

        # Establish a connection to the S3 service
        s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

        try:
            # List all objects in the specified prefix
            objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

            # Download each CSV file in the prefix
            for obj in objects.get('Contents', []):
                key = obj['Key']
                #print(str(ID))
                newID = ID[7:-2] + ID[-1:]
                #print(newID)
                if ('1-1-' + newID) in key:
                    if key.endswith('.csv'):
                        file_name = os.path.join(local_directory, os.path.basename(key))
                        s3.download_file(bucket_name, key, file_name)
                        #print("key: " + str(key))
                        #print("Bucket name: " + str(bucket_name))
                        #print("file name: " + str(file_name))
                        print(f"Downloaded: {file_name}")

            print("All CSV files downloaded successfully.")
        except Exception as e:
            print(f"Error downloading CSV files: {e}")


    #SHows the data page
    def show_data(self):
        self.central_widget.setCurrentWidget(self.dataPage)

    def handler(self, signal_received, frame):
        global zed
        print("inside handlwer")
        zed.disable_recording()
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()
        sys.exit(0)
 
    

    def save_joint_positions(self,body_data):
        global joint_positions
        for body in body_data.body_list:
            if body.id == 0:  # Assuming you are tracking only one person
                joints_to_track = {
                    'left_elbow': 7, 'right_elbow': 8,
                    'left_wrist': 9, 'right_wrist': 10,
                    'left_knee': 13, 'right_knee': 14,
                    'left_ankle': 15,'right_ankle': 16
                }
                # Dictionary to store positions of tracked joints
                tracked_joint_positions = {}
                for joint_name, joint_index in joints_to_track.items():
                    if len(body.skeleton.joints) > joint_index:
                        # Get the 3D position of the joint
                        joint_position = body.skeleton.joints[joint_index].position
                        # Store the position in the dictionary
                        tracked_joint_positions[joint_name] = joint_position
                    else:
                        print(f"Joint {joint_name} not found for body {body.id}")
                # Append the positions to the list
                joint_positions.append(tracked_joint_positions)
        print("Shape of joints = ",np.array(joint_position).shape)

    def parse_args(self, init):
        with open(r"D:\GUI_data\temp.txt", 'r') as f:#temp.txt is where the id for the participant is stored
            ID = f.read()
        init_params = argparse.Namespace(
            input_svo_file='', 
            ip_address='', 
            resolution='', 
            output_svo_file=rf"D:\New Data PEACE Experiment\{ID}\video_1.svo"
        )

        
        if len(init_params.input_svo_file) > 0 and init_params.input_svo_file.endswith(".svo"):
            init.set_from_svo_file(init_params.input_svo_file)
            print("[Sample] Using SVO File input: {0}".format(init_params.input_svo_file))
        elif len(init_params.ip_address) > 0:
            ip_str = init_params.ip_address
            if ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4 and len(
                    ip_str.split(':')) == 2:
                init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
                print("[Sample] Using Stream input, IP : ", ip_str)
            elif ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4:
                init.set_from_stream(ip_str)
                print("[Sample] Using Stream input, IP : ", ip_str)
            else:
                print("Unvalid IP format. Using live stream")
        if "HD2K" in init_params.resolution:
            init.camera_resolution = sl.RESOLUTION.HD2K
            print("[Sample] Using Camera in resolution HD2K")
        elif "HD1200" in init_params.resolution:
            init.camera_resolution = sl.RESOLUTION.HD1200
            print("[Sample] Using Camera in resolution HD1200")
        elif "HD1080" in init_params.resolution:
            init.camera_resolution = sl.RESOLUTION.HD1080
            print("[Sample] Using Camera in resolution HD1080")
        elif "HD720" in init_params.resolution:
            init.camera_resolution = sl.RESOLUTION.HD720
            print("[Sample] Using Camera in resolution HD720")
        elif "SVGA" in init_params.resolution:
            init.camera_resolution = sl.RESOLUTION.SVGA
            print("[Sample] Using Camera in resolution SVGA")
        elif "VGA" in init_params.resolution:
            init.camera_resolution = sl.RESOLUTION.VGA
            print("[Sample] Using Camera in resolution VGA")
        elif len(init_params.resolution) > 0:
            print("[Sample] No valid resolution entered. Using default")
        else:
            print("[Sample] Using default resolution")


    def depth_to_world_coordinates(self,depth, x, y, fx, fy, cx, cy):
    
        normalized_x = (x - cx) / fx
        normalized_y = (y - cy) / fy
        world_x = float(depth) * normalized_x
        world_y = float(depth) * normalized_y
        world_z = float(depth)
        
        return world_x, world_y, world_z

    #thread to record camera to avoid errors
    def record_Camera_Loop(self):
        cameraThread = threading.Thread(target=self.record_camera)
        cameraThread.start()
        signal(SIGINT, gui.handler)
    

    def record_camera(self):
        with open(r"D:\GUI_data\temp.txt", 'r') as f:#temp.txt is where the id for the participant is stored
            ID = f.read()

        print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

        # Create a Camera object
        zed = sl.Camera()
        kps = []
        depths_ = []
        wp = []
        frame_count = 0
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_minimum_distance = 0
        self.parse_args(init_params)
        

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Enable Positional tracking (mandatory for object detection)
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        positional_tracking_parameters.set_as_static = True
        zed.enable_positional_tracking(positional_tracking_parameters)
        
        body_param = sl.BodyTrackingParameters()
        body_param.enable_tracking = True                # Track people across images flow
        body_param.enable_body_fitting = True            # Smooth skeleton move
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
        body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use

        camera_matrix_left = np.load(r'C:\Users\Neurotech\Downloads\camera_matrix_left.npy')

        fx, fy = camera_matrix_left[0,0], camera_matrix_left[1,1]
        cx, cy = camera_matrix_left[0,2], camera_matrix_left[1,2]
        # print("fx = ",camera_matrix_left)
        
        zed.enable_body_tracking(body_param)

        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        body_runtime_param.detection_confidence_threshold = 40

        os.makedirs(os.path.dirname(opt.output_svo_file), exist_ok=True)

        recording_param = sl.RecordingParameters(opt.output_svo_file, sl.SVO_COMPRESSION_MODE.H264) # Enable recording with the filename specified in argument

        err = zed.enable_recording(recording_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Recording ZED : ", err)
            exit(1)

        print("SVO is Recording, use Ctrl-C to stop.")

        frames_data = []
        camera_info = zed.get_camera_information()
        display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
        image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                    , display_resolution.height / camera_info.camera_configuration.resolution.height]


        viewer = gl.GLViewer()
        viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
        bodies = sl.Bodies()
        image = sl.Mat()
        key_wait = 10 
        WP__ = np.empty((18,3,1))

        while viewer.is_available():
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                frame_count += 1
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                zed.retrieve_bodies(bodies, body_runtime_param)

                if bodies.body_list: 
                    if bodies.body_list[0].keypoint_2d is not None:
                        joint_kps = bodies.body_list[0].keypoint_2d
                        kps.append(joint_kps)
                        depth_map = sl.Mat()
                        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

                        wp_array = np.zeros((18,3))
                        
                        for k, (x, y) in enumerate(joint_kps):
                            
                            if 0 <= x < depth_map.get_width() and 0 <= y < depth_map.get_height():
                                err, depth = depth_map.get_value(x, y)
                                if not np.isnan(depth):
                                    world_coordinates = self.depth_to_world_coordinates(depth, x, y, fx, fy, cx, cy)
                                    wp.append(world_coordinates)
                                    depths_.append(depth)
                                    wp_array[k] = world_coordinates
                                    
                        WP__ = np.dstack((WP__, wp_array))

                viewer.update_view(image, bodies) 
                image_left_ocv = image.get_data()
                cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
                cv2.imshow("ZED | 2D View", image_left_ocv)
                key = cv2.waitKey(key_wait)
                if key == 113: # for 'q' key
                    print("Exiting...")
                    if joint_kps is not None:
                        np.save(fr"D:\New Data PEACE Experiment\{ID}\{ID}_joint_kps.npy", joint_kps)
                        np.save(fr"D:\New Data PEACE Experiment\{ID}\{ID}_joint_depths.npy", depths_)
                        np.save(fr"D:\New Data PEACE Experiment\{ID}\{ID}_world_points.npy", WP__)
                    break
                if key == 109: # for 'm' key
                    if (key_wait>0):
                        print("Pause")
                        key_wait = 0 
                    else : 
                        print("Restart")
                        key_wait = 10 
        viewer.exit()
        image.free(sl.MEM.CPU)
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()
        cv2.destroyAllWindows()

        return frames_data
    


    #converting .svo into .avi
    class AppType(enum.Enum):
        LEFT_AND_RIGHT = 1
        LEFT_AND_DEPTH = 2
        LEFT_AND_DEPTH_16 = 3


    def progress_bar(self, percent_done, bar_length=50):
        #Display a progress bar
        done_length = int(bar_length * percent_done / 100)
        bar = '=' * done_length + '-' * (bar_length - done_length)
        sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
        sys.stdout.flush()

    def convert_Camera_Loop(self):
        cameraThread = threading.Thread(target=self.converting)
        cameraThread.start()

    def converting(self):
        svo_input_path = opt1.input_svo_file
        output_dir = opt1.output_path_dir
        avi_output_path = opt1.output_avi_file 
        output_as_video = True    
        app_type = self.AppType.LEFT_AND_RIGHT
        if opt1.mode == 1 or opt1.mode == 3:
            app_type = self.AppType.LEFT_AND_DEPTH
        if opt1.mode == 4:
            app_type = self.AppType.LEFT_AND_DEPTH_16
        
        # Check if exporting to AVI or SEQUENCE
        if opt1.mode !=0 and opt1.mode !=1:
            output_as_video = False

        if not output_as_video and not os.path.isdir(output_dir):
            sys.stdout.write("Input directory doesn't exist. Check permissions or create it.\n",
                            output_dir, "\n")
            exit()

        # Specify SVO path parameter
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_input_path)
        init_params.svo_real_time_mode = False  # Don't convert in realtime
        init_params.coordinate_units = sl.UNIT.METER  # Use milliliter units (for depth measurements)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Set configuration parameters for the ZED
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.depth_minimum_distance = 0

        # Create ZED objects
        zed = sl.Camera()
        # print("aat alay")
        # Open the SVO file specified as a parameter
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            sys.stdout.write(repr(err))
            zed.close()
            exit()
        
        # Get image size
        image_size = zed.get_camera_information().camera_configuration.resolution
        width = image_size.width
        height = image_size.height
        width_sbs = width * 2
        
        # Prepare side by side image container equivalent to CV_8UC4
        svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

        # Prepare single image containers
        left_image = sl.Mat()
        right_image = sl.Mat()
        depth_image = sl.Mat()

        video_writer = None
        if output_as_video:
            # Create video writer with MPEG-4 part 2 codec
            video_writer = cv2.VideoWriter(avi_output_path,
                                        cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                        zed.get_camera_information().camera_configuration.fps,
                                        (width_sbs, height))
            if not video_writer.isOpened():
                sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
                                "permissions.\n")
                zed.close()
                exit()
        
        rt_param = sl.RuntimeParameters()
        rt_param.enable_fill_mode = True

        # Start SVO conversion to AVI/SEQUENCE
        sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

        nb_frames = zed.get_svo_number_of_frames()

        while True:
            err = zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                svo_position = zed.get_svo_position()

                # Retrieve SVO images
                zed.retrieve_image(left_image, sl.VIEW.LEFT)

                if app_type == self.AppType.LEFT_AND_RIGHT:
                    zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                elif app_type == self.AppType.LEFT_AND_DEPTH:
                    zed.retrieve_image(right_image, sl.VIEW.DEPTH)
                elif app_type == self.AppType.LEFT_AND_DEPTH_16:
                    zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

                if output_as_video:
                    # Copy the left image to the left side of SBS image
                    svo_image_sbs_rgba[0:height, 0:width, :] = left_image.get_data()

                    # Copy the right image to the right side of SBS image
                    svo_image_sbs_rgba[0:, width:, :] = right_image.get_data()

                    # Convert SVO image from RGBA to RGB
                    ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba, cv2.COLOR_RGBA2RGB)

                    # Write the RGB image in the video
                    video_writer.write(ocv_image_sbs_rgb)
                else:
                    # Generate file names
                    filename1 = output_dir +"/"+ ("left%s.png" % str(svo_position).zfill(6))
                    filename2 = output_dir +"/"+ (("right%s.png" if app_type == self.AppType.LEFT_AND_RIGHT
                                            else "depth%s.png") % str(svo_position).zfill(6))
                    # Save Left images
                    cv2.imwrite(str(filename1), left_image.get_data())

                    if app_type != self.AppType.LEFT_AND_DEPTH_16:
                        # Save right images
                        cv2.imwrite(str(filename2), right_image.get_data())
                    else:
                        # Save depth images (convert to uint16)
                        cv2.imwrite(str(filename2), depth_image.get_data().astype(np.uint16))

                # Display progress
                self.progress_bar((svo_position + 1) / nb_frames * 100, 30)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                self.progress_bar(100 , 30)
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
        if output_as_video:
            # Close the video writer
            video_writer.release()

        zed.close()



    #thread to avoid errors
    def record_EEG_Loop(self):
        EEGthread = threading.Thread(target=self.record_EEG)
        EEGthread.start()

    def record_EEG(self):
        start_time = time.time#start time to record for allotted time

        with open(r"D:\GUI_data\temp.txt", 'r') as f:#get participant ID
            ID = f.read()

        # Define the MAC-address of the acquisition device used in OpenSignals
        mac_address = "00:07:80:4B:18:75"
        # Resolve stream
        print("Looking for an available OpenSignals stream from the specified device...")
        os_stream = resolve_stream("type", mac_address)

        # Create an inlet to receive signal samples from the stream
        inlet = StreamInlet(os_stream[0])
        samples = []
        timestamps = []
        print("EEG Stream found")
        while True:
            # Receive samples
            sample, timestamp = inlet.pull_sample()

            samples.append(sample)
            timestamps.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))#append the time that the data was recorded to match with camera
            #print(time.time)
            elapsed_time = time.time() - start_time
            if elapsed_time >= 900: #Set the length of the data aquisition and record for that long
                break

        dict = {'samples': samples, 'timestamp': timestamps}
        df = pd.DataFrame(dict)#save EEG data and time to a csv
        df.to_csv(rf"D:\New Data PEACE Experiment\{ID}\EEG_Stream.csv", index=False)
        print("All Done")

    def participant_loop(self):#loop to avoid errors
        partThread = threading.Thread(target=self.show_participant)
        partThread.start()

    def show_participant(self):#runs participant stuff
        subprocess.run(["python", r"D:\GUI_data\ID.py"])

    def create_wall_page(self, wall_num):#creates the pages for the walls
        page = QWidget()
        layout = QVBoxLayout()
        for motor_num in range(1, 5):
            motor_label = QLabel(f"Wall {wall_num} Motor {motor_num}")
            slider = QSlider()
            slider.setOrientation(1)
            slider.setRange(0, 15)
            slider.valueChanged.connect(lambda value, wall=wall_num, motor=motor_num: self.on_slider_change(value, wall, motor))
            layout.addWidget(motor_label)
            layout.addWidget(slider)
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.show_home_page)
        layout.addWidget(back_button)
        page.setLayout(layout)
        return page

    def show_wall_page(self, wall_num):
        self.central_widget.setCurrentWidget(self.wall_pages[wall_num - 1])

    def create_roof_page(self):#creates the page for the roof
        page = QWidget()
        layout = QVBoxLayout()

        # Create sliders for the roof motors
        roof_motor_label = QLabel("Roof Motors")
        layout.addWidget(roof_motor_label)

        for wall_num in range(1, 5):
            motor_label = QLabel(f"Roof Motor {wall_num}")
            slider = QSlider()
            slider.setOrientation(0)
            slider.setRange(0, 15)
            # Connect slider value changes to update global_motor_values[:, 4]
            slider.valueChanged.connect(lambda value, wall=wall_num: self.on_slider_change(value,  wall, 4))
            layout.addWidget(motor_label)
            layout.addWidget(slider)

        back_button = QPushButton("Back")
        back_button.clicked.connect(self.show_home_page)
        layout.addWidget(back_button)

        page.setLayout(layout)
        return page

    def show_roof_page(self):
        self.central_widget.setCurrentWidget(self.roof_page)

    def open_ports(self):
        # ----- Create a list of serial objects
        serial_objects = [serial.Serial(port, 9600) for port in serial_ports]

    # ----- Send the park position command to the motors resetting motors
    def park_motors(self):
        # Set backward
        for j in range(0,park_prd):
            for i in range(0, motor_num):
                self.send_motor_command(self.motor_serial[i], self.motor_pin[i], BACKWARD)
            #prv_setup = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            print("Parking motors.")
            time.sleep(sleep_prd)

    def exit_application(self):
        self.park_motors()
        for serial_object in serial_objects:
            serial_object.close()
        QApplication.instance().quit()  # Close the application


    def show_music_page(self):
        self.central_widget.setCurrentWidget(self.music_page)

    def play_music(self, file_path):
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.media_player.play()

    def stop_music(self):
        self.media_player.stop()

    def show_home_page(self):
        self.central_widget.setCurrentWidget(self.page_1)

    #change walls when slider adjusted
    def on_slider_change(self, value, wall_num, motor_num):
        global cur_stat
        self.motor_values[wall_num - 1][motor_num - 1] = value
        self.global_motor_values = np.array(self.motor_values)  # Update global numpy array
        cur_stat = self.global_motor_values
        #print(f"Wall {wall_num} Motor {motor_num} value: {value}")
        #print(self.global_motor_values)

    def reset_sliders(self):
        for wall_num in range (1,5):
            for motor_num in range(1, 5):
                slider = self.wall_pages[wall_num - 1].layout().itemAt(motor_num * 2 - 1).widget()
                slider.setValue(0)  # Reset the slider value to 0
                self.on_slider_change(0, wall_num, motor_num)
        self.park_motors()

    def apply_setup(self, setup):
        global cur_stat
        cur_stat = setup
        for wall_num in range (1,5):
            for motor_num in range(1, 5):
                slider = self.wall_pages[wall_num - 1].layout().itemAt(motor_num * 2 - 1).widget()
                value = setup[wall_num-1][motor_num-1]
                slider.setValue(value)
                self.on_slider_change(value, wall_num, motor_num)

    #sends commands to the arduino
    def send_motor_command(self, serial_number, pin_number, motor_mode):
        # Ensure motor_number is within the valid range (1 to 3).
        motor_number = pin_number
        # Ensure motor_mode is within the valid range (0 to 2).
        motor_mode = motor_mode
        # Combine the motor_number and motor_mode into a single string message.
        message = f"{pin_number},{motor_mode}\n"
        # Send the message as bytes to the appropriate Arduino.
        serial_object = serial_objects[serial_number - 1]
        serial_object.write(bytes(message, 'utf-8'))
        #print(f"Sent command: {message.strip()}")

    #does the math to determine movement of the motors
    def send_motor_control(self):
        global cur_stat
        global prv_setup
        print(cur_stat)
        print(prv_setup)
        cur = cur_stat
        prv = prv_setup
        # find time difference
        diff_array = cur - prv
        diff_array = diff_array.flatten()
        print(diff_array)
        # set visit array
        move_check = np.zeros(motor_num)
        # go over the loop
        # PEACE Control here
        while sum(abs(diff_array) > 0):#Checks if the walls need to move forward or backward and moves accordingly until done
                    for i in range(0, motor_num):
                        if diff_array[i] > 0:
                            if move_check[i]==0:
                                self.send_motor_command(self.motor_serial[i], self.motor_pin[i], FORWARD)
                                move_check[i] = 1
                            diff_array[i] -= 1
                        if diff_array[i] < 0:
                            if move_check[i]==0:
                                self.send_motor_command(self.motor_serial[i], self.motor_pin[i], BACKWARD)
                                move_check[i] = 1
                            diff_array[i] += 1
                    # sleep period
                    time.sleep(sleep_prd)

        for i in range(0, motor_num):
            self.send_motor_command(self.motor_serial[i], self.motor_pin[i], STOP)
        # set the state
        prv_setup = cur_stat
        print("Sending motor command:")
        print(self.global_motor_values.flatten())

if __name__ == "__main__":
    with open(r"D:\GUI_data\temp.txt", 'r') as f:
            ID = f.read()
    app = QApplication(sys.argv)
    gui = MotorControlGUI()
    gui.show()
    signal(SIGINT, gui.handler)
    parser = argparse.ArgumentParser()
    opt = argparse.Namespace(
        input_svo_file='', 
        ip_address='', 
        resolution='', 
        output_svo_file = fr"D:\New Data PEACE Experiment\{ID}\video_1.svo"
    )

    parser1 = argparse.ArgumentParser()
    opt1 = argparse.Namespace( 
        input_svo_file = fr"D:\New Data PEACE Experiment\{ID}\video_1.svo",
        mode = 1, 
        output_path_dir = 'C:/Users/Neurotech/Downloads/',
        output_avi_file = fr"D:\New Data PEACE Experiment\{ID}\video.avi"
    )
    
    
    sys.exit(app.exec_())