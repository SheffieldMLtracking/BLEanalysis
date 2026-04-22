from IPython.display import display, Markdown
from scipy.stats import norm, cauchy, laplace
from matplotlib import pyplot as plt
import numpy as np

from . import BleLog


def rss_diff_at_gamma(ble_log: BleLog, gamma: float, packet_gap=1) -> list[int]:
    rss_at_gamma = ble_log.all_rss_at_gamma(gamma)
    rss_differences = []

    for x in range(len(rss_at_gamma) - packet_gap):
        rss_differences.append(rss_at_gamma[x] - rss_at_gamma[x + packet_gap])

    return rss_differences


def range_trial_rss_diffs_in_blocks(range_trials, packet_gap=1, log=False):
    for start_angle in range(-15, 360 - 15, 30):
        display(Markdown(f"#### {start_angle} to {(start_angle + 30) % 360} Degrees"))
        plt.figure(figsize=(27, 5))
        graph_num = 1
        combined_diffs = []

        for name, ble_log in range_trials.items():
            diffs = []

            for angle in range(start_angle, start_angle + 29):
                diffs.extend(rss_diff_at_gamma(ble_log, angle, packet_gap))

            combined_diffs.extend(diffs)

            plt.subplot(1, 10, graph_num)
            plt.hist(diffs, bins=np.arange(-20, 20))
            plt.yscale("log") if log else None
            plt.title(name)
            graph_num += 1

        # draw combined graph with a laplace & normal mix distribution
        plt.subplot(1, 10, graph_num)
        plt.hist(combined_diffs, bins=np.arange(-20, 20), density=True)
        x_values = np.linspace(-20, 20, 240)

        mix = 0.351 * norm.pdf(x_values, 0, 3.464) + 0.649 * laplace.pdf(x_values, 0, 3.139)
        plt.plot(x_values, mix)

        plt.yscale("log") if log else None
        plt.title("Combined")
        ax = plt.gca()
        ax.set_facecolor("#5E1414")
        plt.show()


def combined_rss_at_gamma_analysis(range_dict, packet_gap):
    range_rss_diffs = []

    for range_log in range_dict.values():
        for degree in range(0, 360 - 1):
            range_rss_diffs.extend(rss_diff_at_gamma(range_log, degree, packet_gap))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(range_rss_diffs, bins=np.arange(-20, 20))
    plt.title("All Range Trial Differences for All Angles")
    plt.ylabel("Frequency")
    plt.xlabel("RSS Difference (dBm)")

    plt.subplot(1, 3, 2)
    plt.hist(range_rss_diffs, bins=np.arange(-20, 20))
    plt.yscale("log")
    plt.title("Log of Differences for All Angles")
    plt.ylabel("Log of Frequency")
    plt.xlabel("Rss Difference (dBm)")

    # normal and cauchy distributions overlaid over data
    plt.subplot(1, 3, 3)
    plt.title("Normal & Cauchy PDF")
    x_vals = np.linspace(-20, 20, 40 * 2)
    plt.plot(x_vals, norm.pdf(x_vals, 0.5, 4), 'orange')
    plt.plot(x_vals, cauchy.pdf(x_vals, loc=0.5, scale=2.124), 'red')
    plt.plot(x_vals, 0.351 * norm.pdf(x_vals, 0.5, 3.464) + 0.649 * laplace.pdf(x_vals, 0.5, 3.139), 'purple')
    plt.hist(range_rss_diffs, bins=np.arange(-20, 20), density=True)
    plt.show()

    return range_rss_diffs