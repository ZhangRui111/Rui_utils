# import matplotlib as mlp
# Matplotlib is currently using agg, which is a non-GUI backend.
# mlp.use('Agg')
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

    plt.ion()
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


def aspect_ratio_image(ratio):
    """
    Change the aspect ratio of multiple image (or axes).
    :return:
    """
    # Refer to https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    # for Matplotlib Style Gallery
    plt.style.use('ggplot')
    x = np.linspace(-5, 5, 100)
    y1 = np.exp(0.8 * x)
    y2 = np.sin(x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y1)
    ax.plot(x, y2)

    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()

    # The abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    # Or we can utilise the get_data_ratio method which is more concise
    # ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    plt.show()


def aspect_ratio_multi_image(ratio):
    """
    Change the aspect ratio of multiple image (or axes)
    so that they have exactly the same display aspect ratio.
    :return:
    """
    y1 = np.random.uniform(0.1, 0.7, size=(167,))
    y2 = np.random.uniform(1, 100, size=(167,))
    y1 = sorted(y1)
    y2 = sorted(y2)

    fig = plt.figure(figsize=(10, 3))
    # fig.set_edgecolor('red')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharex=ax1)

    ax1.plot(y1)
    ax1.set_ylim([min(y1) * 0.9, max(y1) * 1.1])
    ax1.set_ylabel('y1')

    ax2.plot(y2)
    ax2.set_ylim([min(y2) * 0.9, max(y2) * 1.1])
    ax2.set_ylabel('y2')

    for ax in [ax1, ax2]:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        print((xmax - xmin) / (ymax - ymin))
        ax.set_aspect(abs((xmax - xmin) / (ymax - ymin)) * ratio, adjustable='box')

    plt.show()


# if __name__ == '__main__':
#     aspect_ratio_image(0.5)
#     aspect_ratio_multi_image(0.5)
