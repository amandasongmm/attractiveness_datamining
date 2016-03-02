import ujson
import numpy as np
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool

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


def mscatter(p, x, y, marker):
    p.scatter(x, y, marker=marker, size=15,
              line_color="navy", fill_color="orange", alpha=0.5)


def mtext(p, x, y, text):
    p.text(x, y, text=[text],
           text_color="firebrick", text_align="center", text_font_size="10pt")


def my_scatter():
    pc_ind, img_path_list = load_related_data()
    output_file("toolbar.html")
    source = ColumnDataSource(
        data=dict(
            x=pc_ind[0, :],
            y=pc_ind[1, :],
            imgs=img_path_list
        )
    )

    hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="42" alt="@imgs" width="42"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@desc</span>
                <span style="font-size: 15px; color: #966;">[$index]</span>
            </div>
            <div>
                <span style="font-size: 15px;">Location</span>
                <span style="font-size: 10px; color: #696;">($x, $y)</span>
            </div>
        </div>
        """
    )

    p = figure(plot_width=1000, plot_height=1000, tools=[hover], title="Mouse over the dots")
    p.circle('x', 'y', size=10, source=source)
    show(p)
    return


# def main():
#     pc_ind, img_path_list = load_related_data()  # pc_ind: 10 * 200
#     img_path_list = np.array(img_path_list)
#     for cur_dim in range(pc_ind.shape[0]):
#         cur_list = pc_ind[cur_dim, :]
#         name_list = img_path_list[cur_list]
#         plist = []
#         for item in name_list[1:10]:
#             l = figure(x_range=(0, 1), y_range=(0, 1))
#             l.image_url(url=[item], x=0, y=1)
#             plist.append(l)
#         p = GridPlot(children=[plist])
#         output_file("grid.html", title="grid.py example")
#     show(p)
#     return


# if __name__ == '__main__':
#     main()
my_scatter()


