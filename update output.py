import csv
import numpy as np
from scipy.interpolate import interp1d


def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_number']) for row in data])
    car_ids = np.array([int(float(row['Car_Id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['Car_Bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['License_Plate_Bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_number'] for p in data if int(float(p['Car_Id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_number'] = str(frame_number)
            row['Car_Id'] = str(car_id)
            row['Car_Bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['License_Plate_Bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['License_Plate_Bbox_Score'] = '0'
                row['License_Number'] = '0'
                row['License_Number_Score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_number']) == frame_number and int(float(p['Car_Id'])) == int(float(car_id))][0]
                row['License_Plate_Bbox_Score'] = original_row['License_Plate_Bbox_Score'] if 'License_Plate_Bbox_Score' in original_row else '0'
                row['License_Number'] = original_row['License_Number'] if 'License_Number' in original_row else '0'
                row['License_Number_Score'] = original_row['License_Number_Score'] if 'License_Number_Score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data


# Load the CSV file
with open('Output/Test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_number', 'Car_Id', 'Car_Bbox', 'License_Plate_Bbox', 'License_Plate_Bbox_Score', 'License_Number', 'License_Number_Score']
with open('Output/Test_Interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
