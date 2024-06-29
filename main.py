# Import Library
import cv2
import matplotlib as plt
import numpy as np
from ultralytics import YOLO

import utils
from sort.sort import *
from utils import Get_Car, Read_License_Plate, Write_CSV

motion_tracker = Sort()
# Save the Results
Results = {}

# Load Models
Model_Yolo8 = YOLO('Models/yolov8n.pt')
License_Plate_Detector_Model = YOLO('Models/Model Deteksi Plat Kendaraan.pt')

# Load Video
cap = cv2.VideoCapture('Video Lalu Lintas.mp4')

# Kind Of Vehicles
Class_Vehicles = [2, 3, 5, 7]

# Read Frames (Membaca Frame)
frame_number = -1
ret = True
while ret:
    frame_number += 1
    ret, frame = cap.read()
    if ret:
        Results[frame_number] = {}
        # Vehicles Detection (Deteksi Kendaraan)
        Detector_Vehicles = Model_Yolo8(frame)[0]
        Detections_Vehicles = []
        for Detection in Detector_Vehicles.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = Detection
            if int(class_id) in Class_Vehicles:
                Detections_Vehicles.append([x1, y1, x2, y2, score])

        # Track Vehicles when the Vehicles moving
        Id_Tracker = motion_tracker.update(np.asarray(Detections_Vehicles))

        # License Plate Detection
        Detector_License_Plate = License_Plate_Detector_Model(frame)[0]
        for Detection_License_Plate in Detector_License_Plate.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = Detection_License_Plate

            # Assign License Plate to Car (Mencari Plat Sesuai dengan Kendaraan)
            XCar1, YCar1, XCar2, YCar2, Car_Id = Get_Car(Detection_License_Plate, Id_Tracker)

            if Car_Id != -1:
                # Crop License Plate
                Crop_License_Plate = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process License Plate
                Color_Crop_License_Plate = cv2.cvtColor(Crop_License_Plate, cv2.COLOR_BGR2GRAY)
                _, Threshold_Crop_License_Plate = cv2.threshold(Color_Crop_License_Plate, 64, 255, cv2.THRESH_BINARY_INV)
                # Read License Plate Number
                License_Plate_Text, License_Plate_Text_Score = Read_License_Plate(Threshold_Crop_License_Plate)

                if License_Plate_Text is not None:
                    Results[frame_number][Car_Id] = {'car': {'bbox': [XCar1, YCar1, XCar2, YCar2]},
                                                     'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': License_Plate_Text,
                                                                        'bbox_score': score,
                                                                        'text_score': License_Plate_Text_Score}}

# Result
Write_CSV(Results, 'Output/Test.csv')