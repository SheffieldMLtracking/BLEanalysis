from BleFinley.gps_logs import *
from BleFinley.transmitter import Transmitter


class BleLog:
    """
    Stores all the BLE packets received from a specific transmitter.
    Parses the data directly from a log file produced by the tag
    """
    def __init__(self, log_file_path :str, transmitter :Transmitter, gps_log :GpsLog):
        self.transmitter = transmitter
        self.packets = []
        self.gps_log = gps_log

        self.__parse_log(log_file_path)
        self.gps_log.set_origin(self.transmitter.easting, self.transmitter.northing, self.transmitter.altitude)

        print(f"BleLog | tx: {self.transmitter.tx_id}, no.packets: {len(self.packets)}, time: {self.packets[-1].time:.3f}s")

    def __parse_log(self, file_path :str):
        """
        Turns a raw log file from a BLE tag into a list of [BlePacket] objects
        with normalised time stamps
        :rtype: list
        """
        with open(file_path, 'r') as log_file:
            log_file.readline() # ignore the first line
            unsplit_log_data = log_file.read()
            log_data = unsplit_log_data[54:].split('\n')

        # set time offset as timestamp of first ever (valid) packet
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