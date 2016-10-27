from sklearn.utils import shuffle
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2 as cv
import PIL
from PIL import Image
import math
import pandas as pd
import time

__author__ = 'amanda'


def gen_select_df():
    psycho_excel_path = '../MIT2kFaceDataset/Full Attribute Scores/psychology attributes/psychology-attributes.xlsx'
    df = pd.read_excel(open(psycho_excel_path, 'rb'), sheetname='Final Values')

    keep_list = ['Filename', 'attractive', 'caring', 'aggressive', 'kind', 'sociable', 'happy', 'mean', 'responsible',
                 'cold', 'trustworthy', 'friendly']
    select_df = df[keep_list]
    select_df.to_pickle('file_source/keep_list')
    return


def gen_resized_dataset():
    # resize every image into (256, 256)
    im_root_dir = '../MIT2kFaceDataset/2kfaces/'
    save_dir = '/home/amanda/950/2k_resized/'
    df = pd.read_pickle('file_source/keep_list')

    for im_name in df['Filename']:
        im_name = str(im_name)
        img = Image.open(im_root_dir+im_name)
        img = img.resize((256, 256), PIL.Image.ANTIALIAS)
        img.save(save_dir+im_name)
    return


def gen_file_lst():
    # im_root_dir = '~/950/2k_resized/'
    train_ratio = 0.7

    df = pd.read_pickle('file_source/keep_list')

    total_data_num = df.shape[0]
    train_data_num = int(train_ratio * total_data_num)

    # generate training data lst
    lst1 = open('file_source/train.lst', 'wb')
    for i in range(train_data_num):
        one_line = '{}\t{:.6f}\t{}\n'.format(i, df['sociable'].ix[i], df['Filename'].ix[i])
        lst1.write(one_line)

    # generate test data lst
    lst2 = open('file_source/test.lst', 'wb')
    for i, ind in enumerate(range(train_data_num, total_data_num)):
        one_line = '{}\t{:.6f}\t{}\n'.format(i, df['sociable'].ix[ind], df['Filename'].ix[ind])
        lst2.write(one_line)
    return


# To generate a rec file from the image lst, run this command directly on terminal
# ~/950/Github/mxnet/bin/im2rec path_of_lst.lst root_dir_of_imgs save_path.rec

gen_resized_dataset()
