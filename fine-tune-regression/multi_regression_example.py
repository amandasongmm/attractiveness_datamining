import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import logging
import time

# define a network symbol, equivalent to linear regression
net = mx.symbol.Variable('data')
net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=1)
net = mx.symbol.LinearRegressionOutput(data=net, name='lr')

logging.basicConfig(level=logging.DEBUG)


def build_network():
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type='relu')
    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.symbol.Activatio(data=fc2, name='relu2', act_type='relu')
    fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)
    sm1 = mx.symbol.SoftmaxOutput(data=fc3, name='softmax1')
    sm2 = mx.symbol.SoftmaxOutput(data=fc3, name='softmax2')
    softmax = mx.symbol.Group([sm1, sm2])
    return softmax


class Multi_mnist_iterator(mx.io.DataIter):

    def __init__(self, data_iter):
        super(Multi_mnist_iterator, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data


    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_data[0]
        # Different labels should be used here for actual application
        return [('softmax1_label', provide_label[1]), ('softmax2_label', provide_label[1])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]

        return mx.io.DataBatch(data=batch.data, label=[label, label],
                               pad=batch.pad, index=batch.index)


class Multi_accuracy(mx.metric.EvalMetric):

    def __init__(self, num=None):
        super(Multi_accuracy, self).__init__('multi-accuracy', num)

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num != None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)


        if i == None:
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)
        else:
            self.sum_metric[i] += (pred_label.flat == label.flat).sum()
            self.num_inst[i] += len(pred_label.flat)

batch_size = 100
num_epoches = 100
device = mx.gpu(0)
lr = 0.01

network = build_network()
train, val = mnist_iterator(batch_size=batch_size, input_shape=(784, ))
train = Multi_mnist_iterator(train)
val = Multi_mnist_iterator(val)

model = mx.model.FeedForward(
    ctx=device,
    symbol=network,
    num_epoch=num_epoches,
    learning_rate=lr,
    momentum=0.9,
    wd=0.00001,
    initializer=mx.init.Xavier(factor_type='in', magnitude=2.34)
)

model.fit(
    x=train,
    eval_data=val,
    batch_end_callback=mx.callback.Speedometer(batch_size, 50)
)










