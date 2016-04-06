import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.charts import Bar, output_file, show
import pandas as pd
__author__ = 'amanda'


def load_data():
    npz_data = np.load('tmp/LR_model_params.npz')  # Chicago feature
    coefficient = npz_data['coef']
    return coefficient


def plot_coef1(coefficient):
    std_error = np.std(coefficient, axis=0)
    mean_data = np.mean(coefficient, axis=0)

    x_position = np.linspace(1, std_error.shape[0], std_error.shape[0])
    y_position = mean_data

    output_file('Coefficient.html')
    p = figure(title='errorbar', width=1600, height=500)
    p.xaxis.axis_label = 'feature'
    p.yaxis.axis_label = 'Coefficient value'

    df = pd.DataFrame(mean_data)
    dd = pd.read_csv('data/allConfiguralFeatures.csv')
    df['name'] = list(dd.columns.values)[2:]
    df.columns = ['value', 'name']

    p = Bar(df, values='value', agg='mean', title='coefficient', label='name')
    show(p)
    return


def plot_coef2(coefficient):
    std_error = np.std(coefficient, axis=0)
    mean_data = np.mean(coefficient, axis=0)

    x_position = np.linspace(1, std_error.shape[0], std_error.shape[0])
    y_position = mean_data

    output_file('CoefficientErrorBar.html')
    p = figure(title='errorbar', width=1600, height=500)
    p.xaxis.axis_label = 'feature'
    p.yaxis.axis_label = 'Coefficient value'

    p.circle(x_position, y_position, color='red', size=5, line_alpha=0)

    # plot error bars
    err_xs = []
    err_ys = []
    for x, y, yerr in zip(x_position, y_position, std_error):
        err_xs.append((x, x))
        err_ys.append((y - yerr, y + yerr))

    p.multi_line(err_xs, err_ys, color='orange')
    show(p)
    return


def main1():
    coef = load_data()
    plot_coef1(coef)
    plot_coef2(coef)
    return


def main():
    npz_data = np.load('tmp/LR_model_params.npz')  # Chicago feature
    corr_train = npz_data['corr_train']
    corr_test = npz_data['corr_test']
    mae_train = npz_data['mae_train']
    mae_test = npz_data['mae_test']
    return


if __name__ == '__main__':
    main()


