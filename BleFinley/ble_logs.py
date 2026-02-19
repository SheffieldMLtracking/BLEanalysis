from BleFinley.gps_logs import *
from BleFinley.transmitter import Transmitter


class BleLog:
    """
    Stores all the BLE packets received from a specific transmitter.
    Parses the data directly from a log file produced by the tag
    """
    def __init__(self, log_file_path :str, transmitter :Transmitter, gps_log :GpsLog | None):
        """
        :param log_file_path: Path to the log file of a bee tag
        :param transmitter: Only includes packets sent by this transmitter
        :param gps_log: GPS log of the path taken by the bee tag
        """
        self.transmitter = transmitter
        self.packets = []
        self.gps_log = gps_log

        self.__parse_log(log_file_path)

        if gps_log is not None:
            self.gps_log.set_origin(self.transmitter.easting, self.transmitter.northing, self.transmitter.altitude)

        print(f"BleLog | tx: {self.transmitter.tx_id}, no.packets: {len(self.packets)}, time: {self.packets[-1].time:.3f}s")

    def __parse_log(self, file_path :str):
        """
        Turns a raw log file from a BLE tag into a list of [BlePacket] objects
        with normalised time stamps. If no GPS data is given, time stamps are
        normalised from the time of the first packet from the transmitter,
        otherwise it's synchronised with the GPS data
        :rtype: list
        """
        with open(file_path, 'r') as log_file:
            log_file.readline() # ignore the first line
            unsplit_log_data = log_file.read()
            log_data = unsplit_log_data[54:].split('\n')

        # decide how to normalise time stamps
        if self.gps_log is None:
            # set start time as timestamp of first ever (valid) packet
            for data in log_data:
                if len(data) == 29 and data[27:28] == self.transmitter.tx_id:
                    start_time = int(data[14:16] + data[17:19] + data[20:22], 16) * 1e-3
                    break
        else:
            # set start time so gps and ble logs are the same length
            for data in reversed(log_data):
                if len(data) == 29 and data[27:28] == self.transmitter.tx_id:
                    time_offset = int(data[14:16] + data[17:19] + data[20:22], 16) * 1e-3
                    break

            start_time = abs(self.gps_log.total_time() - time_offset)

        for data in log_data:
            if len(data) == 29 and data[27:28] == self.transmitter.tx_id:
                time = (int(data[14:16] + data[17:19] + data[20:22], 16) * 1e-3) - start_time

                if time < 0:
                    continue

                self.packets.append(BlePacket(
                    -int(data[7:9]), # RSS
                    int(int(data[23:25] + data[26:27], 16)), # transmitter angle
                    time
                ))

    def all_rss_at_gamma(self, gamma :float, show_log = False) -> list[int]:
        """
        Returns a list of all the RSS measurements of packets with a given gamma.
        Leaves ~2s between packets to allow the transmitter to do a full rotation.
        :param gamma: Angle, in degrees, stored in the packets
        :param show_log: If True, prints log information
        :returns: List of rss measurements
        """
        current_packet = min(self.packets[:200], key=lambda packet: abs((packet.angle - gamma) % 360))
        current_index = self.packets.index(current_packet)
        rss_values = [current_packet.rss]
        print(f"all_rss_at_gamma | start time: {current_packet.time:.3f}") if show_log else None

        try:
            while True:
                # range of time values of packets ~2s since the previous one
                search_start_time = current_packet.time + 1.75
                search_end_time = current_packet.time + 2.25

                search_space = list(filter(lambda packet: search_start_time <= packet.time <= search_end_time,
                                           self.packets[current_index + 1: current_index + 500]))
                current_packet = min(search_space, key=lambda packet: abs((gamma - packet.angle) % 360))
                current_index = self.packets.index(current_packet)

                rss_values.append(current_packet.rss)
        finally:
            print(f"all_rss_at_gamma | {len(rss_values)} packets found") if show_log else None
            return rss_values

class BlePacket:
    """
    Stores a single BLE packet collected by the bee tags
    """
    def __init__(self, rss :int, angle :float, time :float):
        self.rss = rss
        self.angle = angle
        self.time = time

    def __str__(self):
        return f"BlePacket | Time: {self.time}, RSS: {self.rss}, Angle: {self.angle}"