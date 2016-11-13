import mxnet as mx
import importlib

__author__ = 'amanda'


model = mx.model.FeedForward.load('model_exp/my_model', 10, ctx=mx.gpu(0))
fea_symbol = model.symbol.get_internals()['flatten_output']
fc_new = mx.symbol.FullyConnected(data=fea_symbol, name='fc_last', num_hidden=1)
symbol = mx.symbol.LinearRegressionOutput(data=fc_new, name='softmax')

# If you'd like to see the network structure, run the plot_network function
a = mx.viz.plot_network(symbol=symbol, node_attrs={'shape':'oval','fixedsize':'false'})

a.render('test')  # It will generate a pdf picture of the network.

