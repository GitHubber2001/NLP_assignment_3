import matplotlib.pyplot as plt


def save_plots() -> None:
    """Saves all existing plots"""

    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(f"saved_data/figure{i}.png")


def plot_histogram(data, title="", xlabel="", ylabel="") -> None:
    plt.figure()
    plt.hist(data, "auto")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show(block=False)
