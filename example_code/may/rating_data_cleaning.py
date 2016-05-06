import numpy as np
from functools import reduce
import itertools

__author__ = 'amanda'

'''
The purpose of this code is to generate a clean rating array.
Filter out adversial rater: who gave the same rating to every face.
Filter out the data entries which contains Nan.
If found such a rater, also delete his/ her twin's data. Save the twin index for later reference.
'''


def clean_rating_data():

    full_rating = np.genfromtxt('data/rating_matrix.csv', delimiter='\t')[1:, :]  # 1548 * 201.
    twin_num = int(np.nanmax(full_rating[:, 0])+1)  # Column 0: twin index: 0-773 (twin pair num = 773+1=774)
    delete_list = []  # a list of twin index which needs to be deleted by pairs.
    for i in range(2):  # check the twin 1 of 774 twins, then all the paired twin 2.
        half_data = full_rating[0 + i*twin_num: twin_num + i*twin_num, :]
        nan_ind = np.nonzero(np.isnan(half_data).sum(axis=1))  # return is a tuple
        nan_ind = list(itertools.chain.from_iterable(nan_ind))  # convert tuple to list
        bad_sub_ind,  = np.where(half_data[:, 1:].std(axis=1) == 0)  # np.ndarray.
        # bad sub give the same rating to all faces
        if bad_sub_ind:  # check empty or not. If not empty, append to delete_list
            print bad_sub_ind
            print type(bad_sub_ind)
            delete_list.append(bad_sub_ind)
        if nan_ind:
            print nan_ind
            print type(nan_ind)
            delete_list.append(nan_ind)

    delete_union = reduce(np.union1d, delete_list)  # take out the union of elements in the list. flatten the list.
    delete_twin2_indlist = delete_union + twin_num
    full_delete_list = np.append(delete_union, delete_twin2_indlist)

    full_rating[:, 0] = full_rating[:, 0] + 1  # Now the twin index is from 1 - 774.

    full_rating = np.delete(full_rating, full_delete_list, axis=0)  # delete the problematic entry from full_rating.
    remain_twin_ind = full_rating[:, 0]  # contain the index info of the remaining twins.
    remain_twin_pair_number = twin_num - delete_union.shape[0]
    #  From 1-774, then 1-774, With certain numbers filtered out.

    full_rating = full_rating[:, 1:]

    # Save the data for later direct loading.
    np.savez('tmp/clean_rating_data', full_rating=full_rating, remain_twin_ind=remain_twin_ind,
             remain_twin_pair_number=remain_twin_pair_number)
    return full_rating, remain_twin_ind, remain_twin_pair_number


clean_rating_data()
