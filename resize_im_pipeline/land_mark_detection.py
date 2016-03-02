# This program will first find the landmarks of a face, then use it to crop all the faces.
#
# Build on dlib codes, adapted from Chad's code
#
# __author__ = 'amanda'
# __date__ =  2/26/2016


import os
import dlib
from skimage import io
import pandas as pd
from params import Params


def landmark_comp():
    p = Params()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p.predictor_path)

    if p.visualize == 1:
        win = dlib.image_window()

    landmark_array = pd.DataFrame(columns=['image', 'point', 'x', 'y'])

    # Iterate over images to generate landmarks
    for cur_img_num in range(200):
        print cur_img_num
        im_file = os.path.join(p.faces_folder_path, p.im_paths['locations'][cur_img_num])
        img = io.imread(im_file)

        # Visualize if applicable
        if p.visualize == 1:
            win.clear_overlay()
            win.set_image(img)

        # Detect the bounding box of the face, see if we can directly use the bounding box to crop the face
        box = detector(img, 1)

        # Find the landmarks in the bounding box
        shape = predictor(img, box[0])

        # Visualize the landmarks
        if p.visualize == 1:
            win.add_overlay(shape)
            win.add_overlay(box)
            dlib.hit_enter_to_continue()

        # Add the landmarks to the dataframe
        for cur_lm_ind in range(68):
            x = shape.part(cur_lm_ind).x
            y = shape.part(cur_lm_ind).y
            landmark_array = landmark_array.append(
                pd.Series([str(im_file), cur_lm_ind, x, y], index=['image', 'point', 'x', 'y']), ignore_index=True)

    # save the bounding box
    landmark_array.to_csv('landmark_wo_hairline.csv', index=False)
    print 'Landmarking finished. Data saved in "landmark_wo_hairline.csv"'


def main():
    landmark_comp()

if __name__ == '__main__':
    main()
