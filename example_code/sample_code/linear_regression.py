import numpy as np
from sklearn import preprocessing
from sklearn import linear_model



from sklearn.cross_validation import train_test_split
import ujson
import numpy as np
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool

__author__ = 'amanda'


def load_data():
    npz_data = np.load('tmp/full_rating.npz')
    full_rating = npz_data['full_rating']
    feature_arr = npz_data['feature_arr']

    # Read the 200 image path list
    file_path = './data/imagePathFile.json'
    with open(file_path) as json_file:
        img_path_list = ujson.load(json_file)

    return img_path_list, full_rating, feature_arr


def my_linear():
    img_path_list, full_rating, feature_arr = load_data()  # full_rating = 1545 * 200; feature_arr = 200 * 29.
    feature_arr = preprocessing.scale(feature_arr)
    average_rating = np.mean(full_rating, axis=0)
    clf = linear_model.Ridge(alpha=0)

    clf.fit(feature_arr, average_rating)
    linear_model.Ridge(alpha=0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False,
                       solver='auto', tol=0.001)
    y_predict = clf.predict(feature_arr)
    return y_predict, average_rating


def mscatter(p, x, y, marker):
    p.scatter(x, y, marker=marker, size=15,
              line_color="navy", fill_color="orange", alpha=0.5)


def mtext(p, x, y, text):
    p.text(x, y, text=[text],
           text_color="firebrick", text_align="center", text_font_size="10pt")


def load_related_data():
    # Read the 200 image path list
    file_path = './data/imagePathFile.json'
    with open(file_path) as json_file:
        img_path_list = ujson.load(json_file)

    # Read the pc_ind_list
    npz_data = np.load('tmp/pc_sorted_ind_list.npz')
    pc_ind = npz_data['pc_sorted_ind_list']
    return pc_ind, img_path_list


def my_scatter():
    pc_ind, img_path_list = load_related_data()
    y_predict, average_rating = my_linear()
    output_file("toolbar.html")
    source = ColumnDataSource(
        data=dict(
            x=average_rating,
            y=y_predict,
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


def main():
    my_scatter()


if __name__ == '__main__':
    main()
