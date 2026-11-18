import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_front(front, path):
    fig, ax = plt.subplots()
    if len(front):
        ax.scatter(-front[:, 0], front[:, 1], facecolors='none', edgecolors='tab:blue')
    ax.set(xlabel='Worst-case source distinguishability', ylabel='Installation cost')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
