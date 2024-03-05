import cv2
import sys
import pyzed.sl as sl
import numpy as np
import argparse
from signal import signal, SIGINT
import os
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer


def handler(signal_received, frame):
    global zed
    print("inside handlwer")
    zed.disable_recording()
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    sys.exit(0)

signal(SIGINT, handler)

def save_joint_positions(body_data):
    global joint_positions
    for body in body_data.body_list:
        if body.id == 0: 
            joints_to_track = {
                'left_elbow': 7, 'right_elbow': 8,
                'left_wrist': 9, 'right_wrist': 10,
                'left_knee': 13, 'right_knee': 14,
                'left_ankle': 15,'right_ankle': 16
            }
            tracked_joint_positions = {}
            for joint_name, joint_index in joints_to_track.items():
                if len(body.skeleton.joints) > joint_index:
                    joint_position = body.skeleton.joints[joint_index].position

                    tracked_joint_positions[joint_name] = joint_position
                else:
                    print(f"Joint {joint_name} not found for body {body.id}")
            joint_positions.append(tracked_joint_positions)
    print("Shape of joints = ",np.array(joint_position).shape)


def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")


def depth_to_world_coordinates(depth, x, y, fx, fy, cx, cy):
   
    normalized_x = (x - cx) / fx
    normalized_y = (y - cy) / fy

    # print("Normalized X and Y and depth", type(normalized_x), type(normalized_y), type(depth[1])
    #       ,"\n", depth )
    
    world_x = float(depth[1]) * normalized_x
    world_y = float(depth[1]) * normalized_y
    world_z = float(depth[1])
    
    return world_x, world_y, world_z


def main():
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    zed = sl.Camera()
    kps = []
    depths_ = []
    wp = []
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  
    init_params.coordinate_units = sl.UNIT.METER          
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_minimum_distance = 0
    init_params.depth_maximum_distance = 10
    
    parse_args(init_params)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                
    body_param.enable_body_fitting = True            
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    body_param.body_format = sl.BODY_FORMAT.BODY_18  

    camera_matrix_left = np.load(r'C:\Users\Neurotech\Downloads\camera_matrix_left.npy')

    fx, fy = camera_matrix_left[0,0], camera_matrix_left[1,1]
    cx, cy = camera_matrix_left[0,2], camera_matrix_left[1,2]
    # print("Camera Intrinsic Matrix_left = ",camera_matrix_left)
    
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    os.makedirs(os.path.dirname(opt.output_svo_file), exist_ok=True)

    recording_param = sl.RecordingParameters(opt.output_svo_file, sl.SVO_COMPRESSION_MODE.H264) 

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
    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_bodies(bodies, body_runtime_param)

            
            if bodies.body_list: 
                if bodies.body_list[0].keypoint_2d is not None:
                    # print("flag = ")
                    joint_kps = bodies.body_list[0].keypoint_2d
                    kps.append(joint_kps)
                    # print(np.array(joint_kps))
                    depth_map = sl.Mat()
                    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

                    for x, y in joint_kps:
                        if 0 <= x < depth_map.get_width() and 0 <= y < depth_map.get_height():
                            depth = depth_map.get_value(x, y)

                            if not np.isnan(depth[1]):
                           
                                world_coordinates = depth_to_world_coordinates(depth, x, y, fx, fy, cx, cy)
                                wp.append(world_coordinates)
                                # print("Depth at ({}, {}): {}".format(x, y, world_coordinates))
                                depths_.append(depth[1])


            viewer.update_view(image, bodies) 
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            key = cv2.waitKey(key_wait)
            if key == 113: # for 'q' key
                print("Exiting...")
                if joint_kps is not None:
                    np.save(r'C:\Users\Neurotech\Downloads\joint_kps.npy', joint_kps)
                    np.save(r'C:\Users\Neurotech\Downloads\joint_depths.npy', depths_)
                    np.save(r'C:\Users\Neurotech\Downloads\world_points.npy', wp)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Address, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--output_svo_file', type=str, help='Path to the SVO file that will be written', required= True)
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    frames_data = main()


