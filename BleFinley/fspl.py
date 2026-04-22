from matplotlib import pyplot as plt
import math

from . import BleLog, GpsLog


def signal_loss(distance :float, frequency = 2_400_000_000, transmitter_gain = 14, receiver_gain = 1):
    return 20 * math.log10(distance) + 20 * math.log10(frequency) - 147.55  - transmitter_gain - receiver_gain


def plot_rss_vs_dist_vs_fspl(ble_log :BleLog):
    # plot RSS vs distance
    x_values = []
    y_values = []

    for packet in ble_log.packets:
        y_values.append(packet.rss)

        closest_gps_point = min(ble_log.gps_log.data_points, key=lambda point: abs(point.relative_time - packet.time))
        x_values.append(closest_gps_point.displacement)

    # plot modelled FSPL
    fspl_points = []

    for i in range (1, int(max(x_values))):
        fspl_points.append(-1 * signal_loss(i))

    plt.plot(x_values, y_values, label="Actual RSS")
    plt.plot(fspl_points, label="Expected RSS")
    plt.title(f"RSS vs Distance, FSPL for Transmitter {ble_log.transmitter.tx_id}")
    plt.xlabel("Distance (m)")
    plt.ylabel("RSS (dBm)")
    plt.legend()

def plot_gps_displacement(gps :GpsLog):
    plt.plot(gps.get_displacements())
    plt.ylim(bottom=0)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("GPS Displacement over Time")