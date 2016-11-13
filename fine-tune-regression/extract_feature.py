import mxnet as mx
import numpy as np
import my_util

__author__ = 'amanda'


def archive():
    # output_list = fea_symbol.list_outputs()
    # for i in output_list:
    #     print '%s, \n', i

    # a list of valuable features
    # # stage 1
    # conv1 = fea_symbol['conv_1_output']
    # pool1 = fea_symbol['pool_1_output']
    #
    # # stage 2
    # conv2 = fea_symbol['conv_2_output']
    # pool2 = fea_symbol['pool_2_output']
    #
    # # stage 2: inception 3a,b,c
    #
    # # stage 3: inception 4a,b,c
    # conca_4e = fea_symbol['ch_concat_4e_chconcat_output']
    #
    # # stage 4: inception 5a,b
    # conca_5b = fea_symbol['ch_concat_5b_chconcat_output']
    #
    # # global avg pooling
    # glo_pool = fea_symbol['global_pool_output']
    #
    # flatten = fea_symbol['flatten_output']
    # fc1 = fea_symbol['fc1_output']
    # softmax = fea_symbol['softmax_output']

    # feature_name_list = ['fc1_output']
    # feature_name_list = ['data', 'conv_1_output', 'pool_1_output', 'conv_2_output', 'pool_2_output',
    #                      'ch_concat_4e_chconcat_output', 'ch_concat_5b_chconcat_output',
    #                      'global_pool_output', 'flatten_output', 'fc1_output', 'softmax_output']

    # for feature_name in feature_name_list:
    #     feature_extract(feature_name)

    # def feature_extract(feature_layer_name):
    #     start_t = time.time()
    #     print 'Now computing ', feature_layer_name
    #     feature_extractor = mx.model.FeedForward(ctx=mx.gpu(0),
    #                                              symbol=fea_symbol[feature_layer_name],
    #                                              arg_params=model.arg_params,
    #                                              aux_params=model.aux_params,
    #                                              allow_extra_params=True)
    #     feature_arr = feature_extractor.predict(social_whole_data)
    #     save_path = 'face_feature_extraction/inception_'+feature_layer_name+'.npz'
    #     np.savez(save_path, feature_arr=feature_arr)
    #     print 'Elapsed time', time.time()-start_t, feature_layer_name
    #     print feature_arr.shape
    #     return feature_arr


    # prepare data
    return


def old_workable_version(rec_path, save_path, model_prefix, model_epoch):
    # rec_path = 'file_source/facebook.rec'
    # save_path = 'face_feature_extraction/facebook_photo_inception_flatten.npz'
    # model_prefix = 'inception-bn/Inception-BN'
    # model_epoch = 126
    social_whole_data = my_util.my_data_iter(rec_path=rec_path)

    # load pre-trained inception model
    model = mx.model.FeedForward.load(prefix=model_prefix,
                                      epoch=model_epoch,
                                      ctx=mx.gpu(0))
    fea_symbol = model.symbol.get_internals()
    inter_symbol = fea_symbol['flatten_output']
    internal_feat = mx.model.FeedForward(symbol=inter_symbol,
                                         arg_params=model.arg_params,
                                         aux_params=model.aux_params,
                                         allow_extra_params=True,
                                         ctx=mx.gpu(0))
    feat_output, data, label = internal_feat.predict(social_whole_data, return_data=True)
    print feat_output.shape

    np.savez(save_path, feature_arr=feat_output, im_data=data, social_score=label)
    return


def extract_feature_with_multi_label_dataiter(save_path, path_im_rec, path_img_list, label_width,
                                              model_prefix, model_epoch):
    # save_path = 'face_feature_extraction/inception_flatten_inconsist_label.npz'
    # path_im_rec = 'file_source/incon_feature_whole.rec'
    # path_img_list = 'file_source/incon_feature_whole.lst'
    # model_prefix = 'inception-bn/Inception-BN'
    # epoch = 126
    whole_data_itr = my_util.data_multilabel_iter(path_imrec=path_im_rec,
                                                  path_imglist=path_img_list,
                                                  label_width=label_width)
    model = mx.model.FeedForward.load(prefix=model_prefix,
                                      epoch=model_epoch,
                                      ctx=mx.gpu(0))
    fea_symbol = model.symbol.get_internals()
    inter_symbol = fea_symbol['flatten_output']
    internal_feat = mx.model.FeedForward(symbol=inter_symbol,
                                         arg_params=model.arg_params,
                                         aux_params=model.aux_params,
                                         allow_extra_params=True,
                                         ctx=mx.gpu(0))
    network_feat_output, im_data, multi_label = internal_feat.predict(whole_data_itr, return_data=True)
    print network_feat_output.shape
    np.savez(save_path, feature_arr=network_feat_output, im_data=im_data, multi_label=multi_label)
    return


def visualize_net(rec_path, save_path, model_prefix, model_epoch):
    social_whole_data = my_util.my_data_iter(rec_path=rec_path)

    # load pre-trained inception model
    model = mx.model.FeedForward.load(prefix=model_prefix,
                                      epoch=model_epoch,
                                      ctx=mx.gpu(0))
    fea_symbol = model.symbol.get_internals()
    inter_symbol = fea_symbol['flatten_output']
    internal_feat = mx.model.FeedForward(symbol=inter_symbol,
                                         arg_params=model.arg_params,
                                         aux_params=model.aux_params,
                                         allow_extra_params=True,
                                         ctx=mx.gpu(0))
    feat_output, data, label = internal_feat.predict(social_whole_data, return_data=True)
    print feat_output.shape

    a = mx.viz.plot_network(symbol=inter_symbol, node_attrs={'shape': 'oval', 'fixedsize': 'false'})
    a.render(save_path)  # It will generate a pdf picture of the network.
    return


# old_workable_version(rec_path='file_source/full_feature_whole.rec',
#                      save_path='face_feature_extraction/vgg_localization_flatten_all.npz',
#                      model_prefix='vgg_localization/simple-136-0',
#                      model_epoch=540)

# visualize_net(rec_path='file_source/full_feature_whole.rec',
#               save_path='localization.pdf',
#               model_prefix='vgg_localization/simple-136-0',
#               model_epoch=540)

# extract_feature_with_multi_label_dataiter(save_path='face_feature_extraction/vgg_localization_flatten_consist.npz',
#                                           path_im_rec='file_source/full_feature_whole.rec',
#                                           path_img_list='file_source/full_feature_whole.lst',
#                                           label_width=11,
#                                           model_prefix='vgg_localization/simple-136-0',
#                                           model_epoch=540
#                                           )

extract_feature_with_multi_label_dataiter(save_path='face_feature_extraction/vgg_localization_flatten_inconsist.npz',
                                          path_im_rec='file_source/incon_feature_whole.rec',
                                          path_img_list='file_source/incon_feature_whole.lst',
                                          label_width=29,
                                          model_prefix='vgg_localization/simple-136-0',
                                          model_epoch=540
                                          )
