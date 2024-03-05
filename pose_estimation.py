import torch
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


path = r'/content/drive/MyDrive/Colab Notebooks/MQP EJW - Developing a space for productivity 2023-2024/Data Sets/PEACE room/Main Data Sets - All 3 Settings Completed/PEACEVR02 - Completed/Session 2: Blue, Large, River/camera1.avi'

path2 = r'/content/drive/MyDrive/Colab Notebooks/MQP EJW - Developing a space for productivity 2023-2024/Data Sets/PEACE room/Main Data Sets - All 3 Settings Completed/PEACEVR03 - Completed/Session 4: Blue, Large, Ocean/camera1.avi' #r'/content/drive/MyDrive/Colab Notebooks/MQP EJW - Developing a space for productivity 2023-2024/Data Sets/PEACE room/Main Data Sets - All 3 Settings Completed/PEACEVR06 - Completed/Session 2: Blue, Large, River/camera1.avi'


def get_limbs_from_keypoints(keypoints):
    limb_indices = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3,
        'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7,
        'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11,
        'right_hip': 12, 'left_knee': 13, 'right_knee': 14, 'left_ankle': 15,
        'right_ankle': 16
    }

    limbs = [
        [limb_indices['right_eye'], limb_indices['nose']],
        [limb_indices['right_eye'], limb_indices['right_ear']],
        [limb_indices['left_eye'], limb_indices['nose']],
        [limb_indices['left_eye'], limb_indices['left_ear']],
        [limb_indices['right_shoulder'], limb_indices['right_elbow']],
        [limb_indices['right_elbow'], limb_indices['right_wrist']],
        [limb_indices['left_shoulder'], limb_indices['left_elbow']],
        [limb_indices['left_elbow'], limb_indices['left_wrist']],
        [limb_indices['right_hip'], limb_indices['right_knee']],
        [limb_indices['right_knee'], limb_indices['right_ankle']],
        [limb_indices['left_hip'], limb_indices['left_knee']],
        [limb_indices['left_knee'], limb_indices['left_ankle']],
        [limb_indices['right_shoulder'], limb_indices['left_shoulder']],
        [limb_indices['right_hip'], limb_indices['left_hip']],
        [limb_indices['right_shoulder'], limb_indices['right_hip']],
        [limb_indices['left_shoulder'], limb_indices['left_hip']]
    ]
    return limbs

def pose_estimate(path):
  model = keypointrcnn_resnet50_fpn(pretrained=True)
  model.eval()

  transform = T.Compose([T.ToTensor()])

  video_path = path
  cap = cv2.VideoCapture(video_path)

  output_path = r'/content/drive/MyDrive/Output/output_video.avi'
  fps = 30
  fourcc = cv2.VideoWriter_fourcc(*"XVID")
  out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

  target_second = 1
  cap.set(cv2.CAP_PROP_POS_MSEC, target_second * 1000)

  with torch.no_grad():
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = transform(pil_image).unsqueeze(0)
        prediction = model(image_tensor)
        if prediction[0]['boxes'].shape[0] == 0:
            continue
        if prediction[0]['scores'][0] < 0.7:
            continue
        keypoints = prediction[0]['keypoints'][0].numpy()
        limbs = get_limbs_from_keypoints(keypoints)
        for point in keypoints:
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        for limb in limbs:
            point1 = keypoints[limb[0]]
            point2 = keypoints[limb[1]]
            cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 2)
        if frame_counter % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
            out.write(frame)
            plt.imshow(frame)
            plt.show()

        frame_counter += 1
    cap.release()
    out.release()

  print("Pose estimation completed. Output video saved at:", output_path)

pose_estimate(path2)

