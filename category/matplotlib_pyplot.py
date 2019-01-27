import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_output_plot(data, savepath=None, if_close_figure=True):
    """
    plot numpy array in segment mean value.
    :param data: numpy array.
    :param savepath:  figure save path.
    :param if_close_figure:  whether plt.close()
    :return:
    """
    data_plot = []
    length = data.size
    interval = 250
    size = int(length / interval)
    for i in range(size):
        start = i * interval
        end = (i + 1) * interval
        segment = data[start:end]
        data_plot.append(np.mean(segment))
    x_axis_data = np.arange(0, length, interval)

    plt.plot(x_axis_data, np.asarray(data_plot), label='label')
    plt.title('title')  # plot figure title
    plt.xlabel('xlabel')  # plot figure's x axis name.
    plt.ylabel('ylabel')  # plot figure's y axis name.
    y_axis_ticks = [0, 1000, 2000, 3000, 4000, 5000]  # range of y axis
    plt.yticks(y_axis_ticks)  # set y axis's ticks
    for items in y_axis_ticks:  # plot some lines that vertical to y axis.
        plt.hlines(items, x_axis_data.min(), x_axis_data.max(), colors="#D3D3D3", linestyles="dashed")
    plt.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath + 'data.png')  # save figures.
    plt.show()  # plt.show() must before plt.close()
    if if_close_figure is True:
        plt.close()  # if not close figure, then all plot will be drawn in the same figure.


def show_gray_image(img):
    """
    Show a numpy array ao a gray image.
    :param img: two-dimensional numpy array -- i.e., (210, 160)
    :return:
    """
    plt.imshow(img, cmap="gray")
    plt.show()
    plt.close()
