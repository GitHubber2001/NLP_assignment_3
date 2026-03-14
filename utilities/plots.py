import matplotlib.pyplot as plt


def save_open_plots() -> None:
    """Saves all open plots"""

    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(f"saved_data/figure{i}.png")
