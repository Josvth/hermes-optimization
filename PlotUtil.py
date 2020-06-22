def apply_report_formatting(x_in = 3.2, y_in = 2.4):

    import matplotlib.pyplot as plt

    fig = plt.gcf()

    fig.set_size_inches(x_in, y_in, forward=True)

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.yaxis.label.set_size(8)
    ax.xaxis.label.set_size(8)

    if hasattr(ax, 'zaxis'):
        ax.zaxis.label.set_size(8)

    lg = ax.get_legend()
    if lg is not None:
        plt.setp(lg.get_texts(), fontsize=6)


def apply_report_formatting_single():
    apply_report_formatting(4.7, 3.5)