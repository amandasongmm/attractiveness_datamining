from bokeh.plotting import figure, show, output_file
import ujson

__author__ = 'amanda'


# Read the 200 image path list
file_path = './data/imagePathFile.json'
with open(file_path) as json_file:
    img_path_list = ujson.load(json_file)


def visualize_img(file_path, output_save_name):
    output_file(output_save_name)
    p = figure(x_range=(0, 1), y_range=(0, 1))
    p.image_url(url=[file_path], x=0, y=1)
    show(p)
    return

visualize_img()




