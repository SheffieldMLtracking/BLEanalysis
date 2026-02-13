from convertbng.util import convert_bng
import math
import csv

class GpsLog:
    """
    Stores a single GPS log outputted by a GPS device
    """
    def __init__(self, unix_time, lat, lon, alt, time_offset, start_northing, start_easting, start_alt):
        self.relative_time = unix_time - time_offset
        self.altitude = alt
        self.easting, self.northing = (int(ls[0]) for ls in convert_bng(lon, lat))

        self.displacement = math.sqrt((self.northing - start_northing)**2 + (self.easting - start_easting)**2
                                      + (self.altitude - start_alt)**2)

    def dist_to_point(self, northing :float, easting :float, alt = None) -> float:
        """
        Calculates the distance from the current point to a specified point. Does 2D
        distance if no altitude is specified, otherwise uses three dimensions
        :rtype: float
        """
        if alt is not None:
            return math.sqrt((self.northing - northing)**2 + (self.easting - easting)**2 + (self.altitude - alt)**2)
        else:
            return math.sqrt((self.northing - northing)**2 + (self.easting - easting)**2)

    def __str__(self):
        return (f"Time: {self.relative_time}, Northing: {self.northing}, Easting: {self.easting}, "
                f"Alt: {self.altitude}, Displacement: {self.displacement}")

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
            start_altitude = float(first_row[5])

            print(f"Starting: time {gps_time_offset}, northing {start_northing}, easting {start_easting}, alt: {start_altitude}")

            for row in reader:
                gps_log.append(GpsLog(
                    float(row[2]), float(row[3]), float(row[4]), float(row[5]), gps_time_offset, start_northing, start_easting, start_altitude
                ))

        print(f"GpsLog.parse_csv: time range is {gps_log[-1].relative_time:.3f}s, end displacement: "
              f"{gps_log[-1].displacement:.3f}m, end altitude difference: {gps_log[-1].altitude - start_altitude:.3f}m")

        return gps_log

class BleLog:
    def __init__(self, log_file_path :str, transmitter):
        self.transmitter = transmitter
        self.packets = []

        self.parse_log(log_file_path)

    def parse_log(self, file_path :str):
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
        for data in log_data:
            if len(data) == 29 and data[27:28] == self.transmitter.tx_id:
                time_offset = int(data[14:16] + data[17:19] + data[20:22], 16)
                break

        for data in log_data:
            if len(data) == 29 and data[27:28] == self.transmitter.tx_id:
                self.packets.append(BlePacket(
                    -int(data[7:9]), # RSS
                    int(int(data[23:25] + data[26:27], 16)), # transmitter angle
                    (int(data[14:16] + data[17:19] + data[20:22], 16) - time_offset) * 1e-3
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

class Transmitter:
    """
    Represents a Bluetooth LE transmitter. Stores its coordinates in both longitude & latitude
    and northing & easting format.
    """
    def __init__(self, tx_id :str, longitude :float, latitude :float, altitude :float):
        self.tx_id = tx_id
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.easting, self.northing = (int(ls[0]) for ls in convert_bng(longitude, latitude))

    def easting_northing(self):
        return self.easting, self.northing

    def __str__(self):
        return (f"Transmitter ID: {self.tx_id}; lon, lat: ({self.longitude}, {self.latitude}); altitude: {self.altitude}"
                f"easting, northing: ({self.easting}, {self.northing})")

if __name__ == "__main__":
    tx_a = Transmitter('a', -1.448596, 53.368606, 131.0)
    log_a = BleLog(
        "/home/finley/Git/BLEanalysis/bluetooth_experiments/March 26 2025 Field Trial/straightpath5/straightpath5noperson.log",
        tx_a
    )
    print(len(log_a.packets))
    print(log_a.packets[0])
    print(log_a.packets[-1])