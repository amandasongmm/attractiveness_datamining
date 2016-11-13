import mxnet as mx

__author__ = 'amanda'


def my_data_iter(rec_path, batch_size=2, shuffle=False, rand_mirror=True):
    data_shape = (3, 224, 224)  # [3, height, width]
    data_iter = mx.io.ImageRecordIter(shuffle=shuffle,
                                      path_imgrec=rec_path,
                                      rand_crop=rand_mirror,
                                      rand_mirror=True,
                                      data_shape=data_shape,
                                      batch_size=batch_size,
                                      preprocess_threads=4,
                                      )
    return data_iter

# my_data_iter(rec_path='file_source/sociable_whole.rec')


def data_multilabel_iter(path_imrec, path_imglist, label_width):
    data_iter = mx.io.ImageRecordIter(shuffle=False,
                                      path_imgrec=path_imrec,
                                      data_shape=(3, 224, 224),
                                      label_width=label_width,
                                      rand_mirror=True,
                                      preprocess_threads=4,
                                      batch_size=2,
                                      path_imglist=path_imglist
                                      )
    return data_iter
