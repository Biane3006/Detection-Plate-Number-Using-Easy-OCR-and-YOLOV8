import string
import easyocr

# Initialize the OCR Reader
Reader = easyocr.Reader(['en'], gpu=False)
# Mapping Dictionaries for Character Conversion
Dict_Char_To_Int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}
Dict_Int_To_Char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def Write_CSV(Results, Output_Path):
    with open(Output_Path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_number', 'Car_Id', 'Car_Bbox',
                                                'License_Plate_Bbox', 'License_Plate_Bbox_Score',
                                                'License_Number', 'License_Number_Score'))
        for frame_number in Results.keys():
            for Car_Id in Results[frame_number].keys():
                print(Results[frame_number][Car_Id])
                if 'car' in Results[frame_number][Car_Id].keys() and 'license_plate' in Results[frame_number][Car_Id].keys() and 'text' in Results[frame_number][Car_Id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_number, Car_Id, '[{} {} {} {}]'.format(Results[frame_number][Car_Id]['car']['bbox'][0], Results[frame_number][Car_Id]['car']['bbox'][1], Results[frame_number][Car_Id]['car']['bbox'][2], Results[frame_number][Car_Id]['car']['bbox'][3]), '[{} {} {} {}]'.format(Results[frame_number][Car_Id]['license_plate']['bbox'][0], Results[frame_number][Car_Id]['license_plate']['bbox'][1], Results[frame_number][Car_Id]['license_plate']['bbox'][2], Results[frame_number][Car_Id]['license_plate']['bbox'][3]), Results[frame_number][Car_Id]['license_plate']['bbox_score'], Results[frame_number][Car_Id]['license_plate']['text'], Results[frame_number][Car_Id]['license_plate']['text_score']))
        f.close()

def Check_Format_License_Plate(Text):
    if len(Text) != 7:
        return False
    # Contoh Plat Nomor 'KT 2407 BDN' Ubah Formatnya Nanti
    if  (Text[0] in string.ascii_uppercase or Text[0] in Dict_Int_To_Char.keys()) and \
        (Text[1] in string.ascii_uppercase or Text[1] in Dict_Int_To_Char.keys()) and \
        (Text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or Text[2] in Dict_Char_To_Int.keys()) and \
        (Text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or Text[3] in Dict_Int_To_Char.keys()) and \
        (Text[4] in string.ascii_uppercase or Text[4] in Dict_Int_To_Char.keys()) and \
        (Text[5] in string.ascii_uppercase or Text[5] in Dict_Int_To_Char.keys()) and \
        (Text[6] in string.ascii_uppercase or Text[6] in Dict_Int_To_Char.keys()):
            return True
    else:
        return False

def Format_License(Text):
    License_Plate = ''
    Map_Character_License_Format = {0: Dict_Int_To_Char, 1: Dict_Int_To_Char, 4:Dict_Int_To_Char, 5: Dict_Int_To_Char, 6: Dict_Int_To_Char,
                                    2: Dict_Char_To_Int, 3: Dict_Char_To_Int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if Text[j] in Map_Character_License_Format[j].keys():
            License_Plate += Map_Character_License_Format[j][Text[j]]
        else:
            License_Plate += Text[j]
    return License_Plate

def Read_License_Plate(License_Plate_Crop):
    Detection_Plates = Reader.readtext(License_Plate_Crop)

    for Detection_Plate in Detection_Plates:
        bbox, text, score = Detection_Plate
        text = text.upper().replace(' ', '')
        if Check_Format_License_Plate(text):
            return Format_License(text), score
    return None, None

def Get_Car(Detection_License_Plate, Vehicle_Track_Id):
    x1, y1, x2, y2, score, class_id = Detection_License_Plate

    Found_Car = False

    for j in range(len(Vehicle_Track_Id)):
        XCar1, YCar1, XCar2, YCar2, Car_Id = Vehicle_Track_Id[j]
        if x1 > XCar1 and y1 > YCar1 and x2 < XCar2 and y2 < YCar2:
            Car_Indeks = j
            Found_Car = True
            break
    if Found_Car:
        return Vehicle_Track_Id[Car_Indeks]

    return -1, -1, -1, -1, -1