import os
import cv2
import sys
from os import listdir
from os.path import isfile, join
import PIL
from PIL import Image
import pandas as pd

__author__ = 'amanda'


def crop_faces_from_folder():
    load_dir = '/home/amanda/Dropbox/2015-2016/face_sim_social/facebook_friends/selected_list/'
    im_files = [f for f in listdir(load_dir) if isfile(join(load_dir, f))]
    save_dir = 'facebook_imgs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # crop faces from images.
    for im_path in im_files:
        load_path = load_dir + im_path
        cascPath = 'haarcascade_frontalface_default.xml'

        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)

        # Read the image
        image = cv2.imread(load_path, 1)
        print image.shape

        [im_h, im_w, _] = image.shape
        gray = cv2.imread(load_path, 0)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        print 'Found {0} faces!'.format(len(faces))

        alpha = 0.2
        for (x, y, w, h) in faces:
            x_l = max(int(x-alpha*w), 0)
            y_l = max(int(y-alpha*h), 0)
            x_h = min(int(x+w+alpha*w), im_w)
            y_h = min(int(y+h+alpha*h), im_h)
            # cv2.rectangle(image, (x_l, y_l), (x_h, y_h), (0, 255, 0), 2)
            crop_img = image[y_l:y_h, x_l:x_h, :]
            cv2.imshow('Faces found', crop_img)
            cv2.waitKey(0)
            cv2.imwrite(save_dir+im_path, crop_img)


def resize_ims():
    resize_dir = 'facebook_img_resized/'
    if not os.path.exists(resize_dir):
        os.makedirs(resize_dir)

    load_dir = 'facebook_imgs/'
    im_files = [f for f in listdir(load_dir) if isfile(join(load_dir, f))]

    for im_name in im_files:
        im_name = str(im_name)
        img = Image.open(load_dir+im_name)
        img = img.resize((224, 224), PIL.Image.ANTIALIAS)
        img.save(resize_dir+im_name)
    return


def gen_friend_score_fea_list():
    df = pd.read_csv('Amanda_friend_list - facebook.csv')
    load_dir = 'facebook_img_resized/'
    im_files = [f for f in listdir(load_dir) if isfile(join(load_dir, f))]

    lst1 = open('../file_source/facebook.lst', 'wb')
    print len(im_files)
    for ind, im_file in enumerate(im_files):
        print ind
        friend_num = df.loc[df['image_name'] == im_file, 'friends number'].values[0]
        # friend_num = df[df['image_name'] == im_file]['friends number'].item()
        one_line = '{}\t{}\t{}\n'.format(ind, friend_num, im_file)
        lst1.write(one_line)
    return

gen_friend_score_fea_list()