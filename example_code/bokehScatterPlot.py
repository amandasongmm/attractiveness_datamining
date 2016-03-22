import bokeh.plotting as plt
import numpy as np
import sklearn.manifold as manifold


__author__ = 'amanda'


def load_data():
    data = np.load('tmp/pca_data.npz')
    pca_feature = data['pca_feature_arr']
    return pca_feature


# To clean the image file list and create a new image list.
def create_clean_img_path():

    raw_file_path = './data/sortedImageRawList.xlsx'
    df = pd.read_excel(raw_file_path, sheetname='Sheet1')
    print('Loading done...\n')

    new_paths = []
    for idx, row in df.iterrows():
        new_row = row.str.strip('"')
        new_row = "../imageData/" + new_row
        new_paths.append(new_row)

    save_path = './data/imagePathFile.json'
    with open(save_path, 'w') as f:
        ujson.dump(new_paths, f)


def embed_show(img_list, feats, html_file='figure.html'):
    mapper = manifold.TSNE()
    bad_ind = np.where(np.isnan(np.mean(feats, axis=1)))[0]
    locs = mapper.fit_transform(feats)

    p = plt.figure(plot_width=2400, plot_height=1200)
    p.image_url(x=locs[:, 0], y=locs[:, 1], url=img_list)
    p.circle(x=locs[:, 0], y=locs[:, 1])

    plt.output_file(html_file)
    plt.save(p)





if __name__ == '__main__':
    embed_show(None, None)
