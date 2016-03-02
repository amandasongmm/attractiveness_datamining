import ujson
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import GridPlot
__author__ = 'amanda'


def load_related_data():
    # Read the 200 image path list
    file_path = './data/imagePathFile.json'
    with open(file_path) as json_file:
        img_path_list = ujson.load(json_file)

    # Read the pc_ind_list
    npz_data = np.load('tmp/pc_sorted_ind_list.npz')
    pc_ind = npz_data['pc_sorted_ind_list']
    return pc_ind, img_path_list




def main():
    pc_ind, img_path_list = load_related_data()  # pc_ind: 10 * 200
    img_path_list = np.array(img_path_list)
    for cur_dim in range(pc_ind.shape[0]):
        cur_list = pc_ind[cur_dim, :]
        name_list = img_path_list[cur_list]
        plist = []
        for item in name_list[1:10]:
            l = figure(x_range=(0, 1), y_range=(0, 1))
            l.image_url(url=[item], x=0, y=1)
            plist.append(l)
        p = GridPlot(children=[plist])
        output_file("grid.html", title="grid.py example")
    show(p)
    return


if __name__ == '__main__':
    main()
