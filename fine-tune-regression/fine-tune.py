import logging
import mxnet as mx
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import cv2 as cv
import argparse


__author__ = 'amanda'


def load_data(data_name, batch_size=2, shuffle=False, rand_mirror=True):
    data_shape = (3, 256, 256)  # [3, height, width]
    data_iter = mx.io.ImageRecordIter(shuffle=shuffle,
                                      path_imgrec=data_name,
                                      rand_crop=rand_mirror,
                                      rand_mirror=True,
                                      data_shape=data_shape,
                                      batch_size=batch_size,
                                      preprocess_threads=4,
                                      )
    return data_iter


class FineTune:
    def __init__(self, load_para_num, train_epoch_num, lr, wd):
        self.load_para_num = load_para_num
        self.train_epoch_num = train_epoch_num
        self.learning_rate = lr
        self.train_batch_size = 2
        self.momentum = 0.9
        self.wd = wd
        self.load_model_prefix = 'inception-bn/Inception-BN'
        self.save_model_prefix = 'model_exp/my_model'

    def model_fine_tune(self):
        start_time = time.time()

        train_dataiter = load_data(data_name='./file_source/train.rec')
        test_dataiter = load_data(data_name='./file_source/test.rec')

        model = mx.model.FeedForward.load(self.load_model_prefix, self.load_para_num, ctx=mx.gpu(0))
        fea_symbol = model.symbol.get_internals()['flatten_output']
        fc_new = mx.symbol.FullyConnected(data=fea_symbol, name='fc_last', num_hidden=1)
        symbol = mx.symbol.LinearRegressionOutput(data=fc_new, name='softmax')
        new_model = mx.model.FeedForward(ctx=mx.gpu(),
                                         num_epoch=self.train_epoch_num,
                                         symbol=symbol,
                                         numpy_batch_size=self.train_batch_size,
                                         learning_rate=self.learning_rate,
                                         momentum=self.momentum,
                                         allow_extra_params=True,
                                         initializer=mx.init.Xavier(factor_type='in', magnitude=2.34),
                                         wd=self.wd)

        symbol.save("model_exp/my_model-symbol.json")
        new_model.fit(X=train_dataiter,
                      eval_data=test_dataiter,
                      eval_metric='rmse',
                      batch_end_callback=mx.callback.Speedometer(100, 20),
                      epoch_end_callback=mx.callback.do_checkpoint(self.save_model_prefix))

        print 'Training time for {} epoches is {}\n'.format(self.train_epoch_num, time.time()-start_time)
        return


def train_main():
    load_epoch_num = 126
    train_epoch_num = 1
    lr = 1e-5
    wd = 1e-6
    p = FineTune(load_epoch_num, train_epoch_num, lr, wd)
    logging.basicConfig(level=logging.DEBUG, filename='./model_exp/test.txt')
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

    p.model_fine_tune()
    return


def predict_with_model(rec_file_path, save_file_path):
    print 'Start predicting...\n'
    start_t = time.time()
    # load model
    prefix = 'model_exp/my_model'
    num_round = 10
    model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=2)

    # predict with loaded model
    dataiter = load_data(data_name=rec_file_path)
    outputs, data, label = model.predict(dataiter, return_data=True)

    # save the result
    np.savez(save_file_path, label=label, outputs=outputs)
    print 'Result saved.\n Elapsed time ={}'.format(time.time()-start_t)
    return


def predict_main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    predict_with_model(rec_file_path='./file_source/train.rec', save_file_path='model_visual_evaluation/train1.npz')
    predict_with_model(rec_file_path='./file_source/test.rec', save_file_path='model_visual_evaluation/test1.npz')
    return


predict_main()
# train_main()
