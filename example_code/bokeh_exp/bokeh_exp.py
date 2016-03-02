from bokeh.plotting import figure, output_file, show, save
from bokeh.models import GridPlot
import numpy as np
__author__ = 'amanda'


def test1():
    # prepare some data
    x = range(1, 6)
    y = [6, 7, 2, 4, 5]

    # output to static HTML file
    output_file('isthattrue.html', title='This will appear in the web tag')

    # create a new plot with a title and axis labels
    p = figure(title="this title show on top of an image", x_axis_label='xtest', y_axis_label='ytest')

    # add a line renderer with legend and line thickness
    p.line(x, y, legend='Legend', line_width=10)

    # show the results
    save(p)
    return


def test2():
    N = 50

    x = np.linspace(0, 4*np.pi, N)
    y = np.sin(x)

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,crosshair"

    plots = []
    for i in range(5):
        l = figure(title="line", tools=TOOLS, plot_width=(i+5)*50, plot_height=(i+5)*50)
        l.line(x, y, line_width=i+1, color="gold")
        plots.append(l)


    p = GridPlot(children=[plots])

    output_file("grid.html", title="grid.py example")

    show(p)

    return

test2()