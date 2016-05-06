import numpy as np
import pandas as pd
__author__ = 'amanda'


def import_rater_info():
    path = '../../ratingData/germine_CurrentBiology_twinaesthetics_PUBLIC.csv'
    df = pd.read_csv(path)

    data = np.load('../tmp/clean_rating_data.npz')
    remain_ind = data['remain_twin_ind']
    remain_ind = remain_ind.astype(int)

    def import_one_field(field_name):
        raw = df[field_name]
        if field_name == 'Zygosity.twin1':
            info_ind = []
            for i in raw:
                if i == 'MZ':
                    info_ind.append(0)
                else:
                    info_ind.append(1)
        else:
            info_ind = raw.as_matrix()
        info_ind = np.hstack((info_ind, info_ind))
        info_ind = info_ind[remain_ind]
        return info_ind
    sex_ind = import_one_field('sex_x.twin1')
    age_ind = import_one_field('age.twin1')
    twin_ind = import_one_field('Zygosity.twin1')
    np.savez('../intermediate_metadata/rater_info', sex_ind=sex_ind, age_ind=age_ind, twin_ind=twin_ind)
    return sex_ind, age_ind, twin_ind

import_rater_info()
