import numpy as np
__author__ = 'amanda'


def load_data():
    model_summary = np.load('tmp/plain_lr_chicagoFeature_result.npz')
    model_summary = model_summary['model_summary']
    return model_summary

'''
corr_train, mae_train, mse_train, r_score_train,
corr_test, mae_test, mse_test, r_score_test,
coef_all, coef_back_all
'''


def visualize_linear_model_metrics(data_summary):
    data = {'source': ['training', 'training', 'training', 'training',
                       'testing', 'testing', 'testing', 'testing'],
            'metric_name': ['correlation', 'MAE', 'MSE', 'r2score',
                            'correlation', 'MAE', 'MSE', 'r2score'],
            'value': [np.mean(x) for x in data_summary],
            'data_summary': data_summary}

    bar = Bar(data, values='value', label='metric_name', group='source',
              title='Linear Regression Model Performance', legend='top_right')
    output_file("bar.html")
    show(bar)
    return


def visualize_coef_bar(coef_full):

    # coef_full = coef_full[:, 0:-1]
    num_features = coef_full.shape[1]  # number of data entries
    x = np.arange(num_features)  # the x locations for the groups

    # Prepare data
    y_mean = np.mean(coef_full, axis=0)
    y_std = np.mean(coef_full, axis=0)
    width = 0.5  # bar width

    # Plot
    fig, ax = plt.subplots()
    ax.bar(x, y_mean, width,  color='MediumSlateBlue', yerr=y_std,  error_kw={'ecolor': 'Tomato', 'linewidth': 2})
    axes = plt.gca()
    axes.set_ylim([-0.5, 0.5])
    ax.set_xticks(x+width)
    # ax.set_xticklabels(('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'))
    plt.show()

    return

model_stats = load_data()
a = 12