from convertbng.util import convert_bng
import math
from BLEanalysis.signals import Signals
import csv
import numpy as np

class GpsLog:
    """
    Stores a single GPS log outputted by a GPS device
    """
    def __init__(self, unix_time, lat, lon, alt, time_offset, start_northing, start_easting):
        self.relative_time = unix_time - time_offset
        self.altitude = alt
        self.easting, self.northing = (ls[0] for ls in convert_bng(lon, lat))

        self.displacement = math.sqrt((self.northing - start_northing)**2 + (self.easting - start_easting)**2)

    def __str__(self):
        return f"Time: {self.relative_time}, Northing: {self.northing}, Easting: {self.easting}, Alt: {self.altitude}, Displacement: {self.displacement}"

    @staticmethod
    def parse_csv(file_path :str):
        """
        Turns a CSV file of GPS data into an array of [GpsLog]
        data with normalised time stamps
        :param file_path: File path of the CSV file
        :rtype: list
        """
        gps_log = []

        with open(file_path, 'r') as csvfile:
            next(csvfile)
            reader = csv.reader(csvfile, delimiter=',')

            # set the starting position etc.
            first_row = next(reader)
            gps_time_offset = float(first_row[2])
            start_easting, start_northing = (ls[0] for ls in convert_bng(float(first_row[4]), float(first_row[3])))
            print(f"Starting: time {gps_time_offset}, northing {start_northing}, easting {start_easting}")

            for row in reader:
                gps_log.append(GpsLog(
                    float(row[2]), float(row[3]), float(row[4]), float(row[5]), gps_time_offset, start_northing, start_easting
                ))

        return gps_log

class BlePacket:
    """
    Stores a single BLE packet collected by the bee tags
    """
    def __init__(self, rss :float, transmitter :np.float64, angle :float, time :float, time_offset :float):
        self.rss = rss
        self.transmitter = chr(transmitter.astype(int))
        self.angle = angle
        self.time = (time - time_offset) * 1e-3

    def __str__(self):
        return f"Time: {self.time}, RSS: {self.rss}, ID: {self.transmitter}, Angle: {self.angle}"

    @staticmethod
    def parse_log(file_path :str, transmitter :str, start = 200, end = 500):
        """
        Turns a raw log file from a BLE tag into a list of [BlePacket] objects
        with normalised time stamps. Removes the first and last few packets
        :rtype: list
        """
        packets = Signals(file_path, [transmitter] ,filetype='log',angleOffset=0).data[start:-end]

        time_offset = packets[0, 3]
        parsed_packets = []
        print(f"Starting: time {time_offset}, RSS: {packets[0, 0]}")

        for pkt in packets:
            parsed_packets.append(BlePacket(pkt[0], pkt[1], pkt[2], pkt[3], time_offset))

        return parsed_packets