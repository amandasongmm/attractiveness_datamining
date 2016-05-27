"""
ChangeCFD.py

This will change the CFD style feature titles to our type.

Author: Chad Atalla
Date:   5/20/2016
"""

import pandas as pd

# Grab CFD features
data = pd.read_csv('../ChicagoFaceDataset/cfd2Data.csv')

newData = pd.DataFrame(columns=('img_num', 'nose_width', 'nose_length', 'lip_thickness', 'face_length', 'eye_height', 'eye_width', 'face_width_prom', 'face_width_mouth', 'forehead_length', 'distance_btw_pupils', 'dist_btw_pupils_top','dist_btw_pupils_lip', 'chin_length', 'length_cheek_to_chin', 'brow_to_hair', 'fWHR', 'face_shape', 'heartshapeness', 'nose_shape', 'lip_fullness', 'eye_shape', 'eye_size', 'upper_head_len', 'midface_len', 'chin_size', 'forehead_height', 'cheek_height', 'cheek_prominence', 'face_roundness'))

newData['img_num'] = data['Target']
newData['nose_width'] = (data['Nose_Width'])
newData['nose_length'] = (data['Nose_Length'])
newData['lip_thickness'] = (data['Lip_Thickness'])
newData['face_length'] = (data['Face_Length'])
newData['eye_height'] = (data['Avg_Eye_Height'])
newData['eye_width'] = (data['Avg_Eye_Width'])
newData['face_width_prom'] = (data['Face_Width_Cheeks'])
newData['face_width_mouth'] = (data['Face_Width_Mouth'])
newData['forehead_length'] = (data['Forehead'])

newData['distance_btw_pupils'] = [x+y for x,y in zip(data['Asymmetry_pupil_top'], data['Asymmetry_pupil_lip'])]
newData['length_cheek_to_chin'] = (data['Cheeks_avg'])
newData['brow_to_hair'] = [(x+y)/2 for x,y in zip(data['Midbrow_Hairline_R'], data['Midbrow_Hairline_L'])]

newData['nose_shape'] = (data['Noseshape'])
newData['dist_btw_pupils_top'] = [(x+y)/2 for x,y in zip(data['Pupil_Top_R'], data['Pupil_Top_L'])]
newData['dist_btw_pupils_lip'] = [(x+y)/2 for x,y in zip(data['Pupil_Lip_R'], data['Pupil_Lip_L'])]
newData['chin_length'] = (data['BottomLip_Chin'])

newData['face_shape'] = (data['Faceshape'])
newData['heartshapeness'] = (data['Heartshapeness'])
newData['lip_fullness'] = (data['LipFullness'])
newData['eye_shape'] = (data['EyeShape'])
newData['eye_size'] = (data['EyeSize'])
newData['upper_head_len'] = (data['UpperHeadLength'])
newData['midface_len'] = (data['MidfaceLength'])
newData['chin_size'] = (data['ChinLength'])
newData['forehead_height'] = (data['ForeheadHeight'])
newData['cheek_height'] = (data['CheekboneHeight'])
newData['cheek_prominence'] = (data['CheekboneProminence'])
newData['face_roundness'] = (data['FaceRoundness'])
newData['fWHR'] = (data['fWHR'])

newData.to_csv('convertedCFDFeatures.csv', index=False)