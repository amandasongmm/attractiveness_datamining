import numpy as np

__author__ = 'amanda'


def group_correlation_comp():

    # Load the rating data
    rating_data = np.load('../tmp/clean_rating_data.npz')
    orig_rating = rating_data['full_rating']   #
    mean = np.mean(orig_rating, axis=0)
    att_average = mean.tolist()

    # Calculate correlation array
    group_corr_array = []
    for i in range(orig_rating.shape[0]):
        cov = np.corrcoef(att_average, orig_rating[i, :])
        group_corr_array.append(cov[0, 1])
    np.savez('../intermediate_metadata/group_correlation', group_corr_array=group_corr_array)
    return group_corr_array

group_correlation_comp()
