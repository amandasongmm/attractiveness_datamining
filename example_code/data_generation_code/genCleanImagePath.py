import pandas as pd
import ujson


__author__ = 'amanda'


def create_clean_img_path():

    raw_file_path = '../data/sortedImageRawList.xlsx'
    df = pd.read_excel(raw_file_path, sheetname='Sheet1')
    print('Loading done...\n')

    new_paths = []
    for idx, row in df.iterrows():
        new_row = row.str.strip('"')
        new_row = "../imageData/" + new_row
        new_paths.append(new_row)

    save_path = '../data/imagePathFile.json'
    with open(save_path, 'w') as f:
        ujson.dump(new_paths, f)
    return


create_clean_img_path()