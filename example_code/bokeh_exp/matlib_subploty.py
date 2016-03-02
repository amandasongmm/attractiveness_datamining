import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ujson


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
    pc_ind, img_path_list = load_related_data()
    img_path_list = np.array(img_path_list)
    subplot_num = 200
    plt.close('all')
    for pc_num in range(pc_ind.shape[0]):
        print 'curPC'+str(pc_num)+'\n'
        for cur_p in range(subplot_num):
            plt.subplot(10, 20, cur_p+1)
            cur_ind = pc_ind[pc_num, cur_p]
            img_link = img_path_list[cur_ind]
            image = mpimg.imread(str(img_link)[3:-2])
            plt.axis('off')
            plt.imshow(image)
        save_name = 'pc_visualization/pc'+str(pc_num)+'.png'
        plt.savefig(save_name, bbox_inches='tight')
    return


if __name__ == '__main__':
    main()

