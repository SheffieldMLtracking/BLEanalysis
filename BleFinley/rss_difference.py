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


def range_trial_rss_diffs_in_blocks(range_trials, packet_gap=1, draw_pdf=True, log=False):
    for start_angle in range(-15, 360 - 15, 30):
        fig, axs = plt.subplots(2, 5, layout='constrained', figsize=(15, 6))
        fig.suptitle(f"Combined Δy for γ from {start_angle} to {(start_angle + 30) % 360} Degrees")
        fig.supylabel("Frequency")
        fig.supxlabel("Δy (dBm)")

        col = 0
        row = 0
        combined_diffs = []

        for name, ble_log in range_trials.items():
            diffs = []

            for angle in range(start_angle, start_angle + 29):
                diffs.extend(rss_diff_at_gamma(ble_log, angle, packet_gap))

            combined_diffs.extend(diffs)

            axs[(row, col)].hist(diffs, bins=np.arange(-20, 20))
            axs[(row, col)].set_yscale("log") if log else None
            axs[(row, col)].set_title(name)

            col = (col + 1) % 5
            row = row + 1 if col == 0 else row

        # draw combined graph with a laplace & normal mix distribution
        axs[(row, col)].hist(combined_diffs, bins=np.arange(-20, 20), density=True)

        if draw_pdf:
            x_values = np.linspace(-20, 20, 240)

            mix = 0.351 * norm.pdf(x_values, 0, 3.464) + 0.649 * laplace.pdf(x_values, 0, 3.139)
            axs[(row, col)].plot(x_values, mix)

        axs[(row, col)].set_yscale("log") if log else None
        axs[(row, col)].set_title("Combined")
        axs[(row, col)].set_facecolor("#5E1414")
        plt.show()


def combined_rss_at_gamma_analysis(range_dict, packet_gap, density=True):
    range_rss_diffs = []

    for range_log in range_dict.values():
        for degree in range(0, 360 - 1):
            range_rss_diffs.extend(rss_diff_at_gamma(range_log, degree, packet_gap))

    num_subplots = 3 if density else 2
    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_subplots, 1)
    plt.hist(range_rss_diffs, bins=np.arange(-20, 20))
    plt.title("All Range Trial Differences for All Angles")
    plt.ylabel("Frequency")
    plt.xlabel("Δy (dBm)")

    plt.subplot(1, num_subplots, 2)
    plt.hist(range_rss_diffs, bins=np.arange(-20, 20))
    plt.yscale("log")
    plt.title("Log of Δy for All Angles")
    plt.ylabel("Log of Frequency")
    plt.xlabel("Δy (dBm)")

    # normal and cauchy distributions overlaid over data
    if density:
        plt.subplot(1, num_subplots, num_subplots)
        x_vals = np.linspace(-20, 20, 40 * 2)
        plt.plot(x_vals, cauchy.pdf(x_vals, loc=0.5, scale=2.124), 'red', label="Cauchy PDF")
        plt.plot(x_vals, 0.351 * norm.pdf(x_vals, 0.5, 3.464) + 0.649 * laplace.pdf(x_vals, 0.5, 3.139), 'Orange',
                 label="Normal PDF")
        plt.hist(range_rss_diffs, bins=np.arange(-20, 20), density=True) #, label="Combined Δy histogram")
        plt.title("Density Histograms and PDFs")
        plt.ylabel("Probability")
        plt.xlabel("Δy (dBm)")
        plt.legend()

    plt.show()

    return range_rss_diffs