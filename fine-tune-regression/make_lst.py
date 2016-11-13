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
    inconsistent_list = ['Filename', 'atypical', 'boring', 'calm', 'common', 'confident', 'egotistic',
                         'emotUnstable', 'forgettable',
                         'intelligent', 'introverted', 'unattractive', 'unemotional', 'unfamiliar', 'unfriendly',
                         'unhappy', 'weird',
                         'emotStable',
                         'emotional', 'familiar', 'humble', 'interesting', 'irresponsible', 'memorable', 'normal',
                         'typical', 'uncertain', 'uncommon', 'unintelligent', 'untrustworthy']
    select_df = df[keep_list]
    select_df.to_pickle('file_source/keep_list')
    inconsist_df = df[inconsistent_list]
    inconsist_df.to_pickle('file_source/inconsist_list')
    return


def gen_resized_dataset():
    # resize every image into (256, 256)
    im_root_dir = '../MIT2kFaceDataset/2kfaces/'
    save_dir = '/home/amanda/950/2k_resized/'
    df = pd.read_pickle('file_source/keep_list')

    for im_name in df['Filename']:
        im_name = str(im_name)
        img = Image.open(im_root_dir+im_name)
        img = img.resize((224, 224), PIL.Image.ANTIALIAS)
        img.save(save_dir+im_name)
    return


def gen_file_lst(feature_name):
    # im_root_dir = '~/950/2k_resized/'
    start_t = time.time()
    train_ratio = 0.7

    df = pd.read_pickle('file_source/keep_list')

    total_data_num = df.shape[0]
    train_data_num = int(train_ratio * total_data_num)

    # generate training data lst
    lst1 = open('file_source/'+feature_name+'_train.lst', 'wb')
    for i in range(train_data_num):
        one_line = '{}\t{:.6f}\t{}\n'.format(i, df[feature_name].ix[i], df['Filename'].ix[i])
        lst1.write(one_line)

    # generate test data lst
    lst2 = open('file_source/'+feature_name+'_test.lst', 'wb')
    for i, ind in enumerate(range(train_data_num, total_data_num)):
        one_line = '{}\t{:.6f}\t{}\n'.format(i, df[feature_name].ix[ind], df['Filename'].ix[ind])
        lst2.write(one_line)
    print 'Done. Elapsed time = {}'.format(time.time()-start_t)
    return


def gen_whole_data_lst(feature_name):
    df = pd.read_pickle('file_source/keep_list')
    lst = open('file_source/' + feature_name + '_whole.lst', 'wb')
    for i in range(len(df)):
        one_line = '{}\t{:.6f}\t{}\n'.format(i, df[feature_name].ix[i], df['Filename'].ix[i])
        lst.write(one_line)
    return


def gen_multi_label_lst():
    # im_root_dir = '~/950/2k_resized/'
    start_t = time.time()
    df = pd.read_pickle('file_source/keep_list')
    label_list = ['attractive', 'caring', 'aggressive', 'kind', 'sociable', 'happy', 'mean', 'responsible',
                 'cold', 'trustworthy', 'friendly']  # 11 features

    # lst = open('file_source/full_feature_whole.lst', 'wb')
    # for i in range(len(df)):
    #     one_line = \
    #         '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n'.\
    #             format(i, df['attractive'].ix[i], df['caring'].ix[i], df['aggressive'].ix[i], df['kind'].ix[i],
    #                    df['sociable'].ix[i], df['happy'].ix[i], df['mean'].ix[i], df['responsible'].ix[i],
    #                    df['cold'].ix[i], df['trustworthy'].ix[i], df['friendly'].ix[i], df['Filename'].ix[i])
    #     lst.write(one_line)

    train_ratio = 0.7
    total_data_num = df.shape[0]
    train_data_num = int(train_ratio * total_data_num)

    # generate training data lst
    # lst1 = open('file_source/full_feature_train.lst', 'wb')
    # for i in range(train_data_num):
    #     one_line = \
    #         '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n'.\
    #             format(i, df['attractive'].ix[i], df['caring'].ix[i], df['aggressive'].ix[i], df['kind'].ix[i],
    #                    df['sociable'].ix[i], df['happy'].ix[i], df['mean'].ix[i], df['responsible'].ix[i],
    #                    df['cold'].ix[i], df['trustworthy'].ix[i], df['friendly'].ix[i], df['Filename'].ix[i])
    #     lst1.write(one_line)

    # generate test data lst
    lst2 = open('file_source/full_feature_test.lst', 'wb')
    for i, ind in enumerate(range(train_data_num, total_data_num)):
        one_line = \
            '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n'.\
                format(i, df['attractive'].ix[ind], df['caring'].ix[ind], df['aggressive'].ix[ind], df['kind'].ix[ind],
                       df['sociable'].ix[ind], df['happy'].ix[ind], df['mean'].ix[ind], df['responsible'].ix[ind],
                       df['cold'].ix[ind], df['trustworthy'].ix[ind], df['friendly'].ix[ind], df['Filename'].ix[ind])
        lst2.write(one_line)
    print 'Done. Elapsed time = {}'.format(time.time()-start_t)
    return


def gen_multi_label_inconsist_lst():

    def my_one_line(im_seq_id, i):
        line = '{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t' \
            '{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t' \
            '{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{}\n'.format(im_seq_id, df['atypical'].ix[i],
                                                                                  df['boring'].ix[i],
                                                                          df['calm'].ix[i], df['common'].ix[i],
                       df['confident'].ix[i], df['egotistic'].ix[i], df['emotUnstable'].ix[i], df['forgettable'].ix[i],
                       df['intelligent'].ix[i], df['introverted'].ix[i], df['unattractive'].ix[i],
                       df['unemotional'].ix[i], df['unfamiliar'].ix[i], df['unfriendly'].ix[i], df['unhappy'].ix[i],
                       df['weird'].ix[i], df['emotStable'].ix[i], df['emotional'].ix[i], df['familiar'].ix[i],
                                                                                  df['humble'].ix[i],
                       df['interesting'].ix[i], df['irresponsible'].ix[i], df['memorable'].ix[i], df['normal'].ix[i],
                       df['typical'].ix[i], df['uncertain'].ix[i], df['uncommon'].ix[i], df['unintelligent'].ix[i],
                       df['untrustworthy'].ix[i], df['Filename'].ix[i])

        return line

    df = pd.read_pickle('file_source/inconsist_list')
    inconsistent_list = ['Filename', 'atypical', 'boring', 'calm', 'common', 'confident', 'egotistic',
                         'emotUnstable', 'forgettable',
                         'intelligent', 'introverted', 'unattractive', 'unemotional', 'unfamiliar', 'unfriendly',
                         'unhappy', 'weird',
                         'emotStable',
                         'emotional', 'familiar', 'humble', 'interesting', 'irresponsible', 'memorable', 'normal',
                         'typical', 'uncertain', 'uncommon', 'unintelligent', 'untrustworthy']

    lst = open('file_source/incon_feature_whole.lst', 'wb')
    for cur_i in range(len(df)):
        one_line = my_one_line(cur_i, cur_i)
        lst.write(one_line)

    train_ratio = 0.7
    total_data_num = df.shape[0]
    train_data_num = int(train_ratio * total_data_num)

    # generate training data lst
    lst1 = open('file_source/incon_feature_train.lst', 'wb')
    for cur_i in range(train_data_num):
        one_line = my_one_line(cur_i, cur_i)
        lst1.write(one_line)

    # generate test data lst
    lst2 = open('file_source/incon_feature_test.lst', 'wb')
    for ind, cur_i in enumerate(range(train_data_num, total_data_num)):
        one_line = my_one_line(ind, cur_i)
        lst2.write(one_line)
    return


# To generate a rec file from the image lst, run this command directly on terminal
# ~/950/Github/mxnet/bin/im2rec path_of_lst.lst root_dir_of_imgs save_path.rec
# im_root_dir = '~/950/2k_resized/'
# ./bin/im2rec image.lst image_root_dir output.bin resize=256 label_width=4


# ~/950/Github/mxnet/bin/im2rec full_feature_whole.lst ~/950/2k_resized/ full_feature_whole.rec label_width=11

gen_multi_label_inconsist_lst()
