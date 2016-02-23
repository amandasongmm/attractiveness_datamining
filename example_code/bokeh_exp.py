from bokeh.plotting import figure, output_file, show, save

__author__ = 'amanda'


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
