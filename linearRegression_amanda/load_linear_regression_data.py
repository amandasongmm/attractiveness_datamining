import pandas as pd
__author__ = 'amanda'


class Params:
    def __init__(self):
        self.visualize = 0  # If set to 1, turn on visualization. If set to 0, turn off it.

        '''Set path'''
        self.predictor_path = './tmp/shape_predictor_68_face_landmarks.dat'
        self.faces_folder_path = '../imageData/'  # track the image paths
        self.im_paths = pd.read_csv('./tmp/imageLocations.csv')
