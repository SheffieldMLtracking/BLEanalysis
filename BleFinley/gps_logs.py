from convertbng.util import convert_bng
import math
import csv

class GpsLog:
    """
    Stores all the GPS points in a GPS log file. Parses the
    data points in from a CSV file.
    """

    def __init__(self, gps_file_path: str):
        self.data_points = []
        self.__parse_csv(gps_file_path)
        self.__origin = (self.data_points[0].easting, self.data_points[0].northing, self.data_points[0].altitude)

        print(f"GpsLog | no. points: {len(self.data_points)}, time: {self.total_time():.3f}")

    def __parse_csv(self, gps_file_path: str):
        """
        Turns a CSV file of GPS data into an array of [GpsLog]
        gps data with time stamps and displacements
        relative to the first data point
        :param gps_file_path: File path of the CSV file
        """
        with open(gps_file_path, 'r') as csvfile:
            next(csvfile)
            reader = csv.reader(csvfile, delimiter=',')

            # set the starting position etc.
            first_row = next(reader)

            gps_time_offset = float(first_row[2])
            start_easting, start_northing = (ls[0] for ls in convert_bng(float(first_row[4]), float(first_row[3])))
            start_altitude = float(first_row[5])

            # add the first log to the data points
            self.data_points.append(
                GpsPoint(float(first_row[2]), float(first_row[3]), float(first_row[4]), float(first_row[5]),
                         gps_time_offset, start_northing, start_easting, start_altitude))

            # turn each CSV line into a GpsData object
            for row in reader:
                self.data_points.append(GpsPoint(
                    float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                    gps_time_offset, start_northing, start_easting, start_altitude
                ))

    def set_origin(self, easting, northing, altitude):
        """
        Updates the origin of the GPS log and recalculates all displacements
        """
        self.__origin = (easting, northing, altitude)

        for point in self.data_points:
            point.set_displacement_from_point(*self.__origin)

    def dist_from_first_point(self, easting, northing, altitude):
        """Calculates the distance from the initial data point to a given point"""
        return math.sqrt(
            (self.data_points[0].easting - easting) ** 2
            + (self.data_points[0].northing - northing) ** 2
            + (self.data_points[0].altitude - altitude) ** 2
        )

    def total_time(self) -> float:
        """How long the GPS log runs from, in seconds"""
        return self.data_points[-1].relative_time

    def max_displacement(self) -> float:
        """The maximum displacement from the set origin"""
        return max(self.data_points, key=lambda point: point.displacement).displacement

    def get_displacements(self) -> list[float]:
        """
        Gets a chronological list of all the displacements of GPS points
        from their set origin
        """
        return [getattr(point, "displacement") for point in self.data_points]

    def get_eastings(self) -> list[int]:
        """
        Gets a chronological list of all the Eastings of GPS points
        from their set origin
        """
        return [getattr(point, "easting") for point in self.data_points]

    def get_northings(self) -> list[int]:
        """
        Gets a chronological list of all the Northings of GPS points
        from their set origin
        """
        return [getattr(point, "northing") for point in self.data_points]

    def get_altitudes(self) -> list[int]:
        """
        Gets a chronological list of all the altitudes of GPS points
        from their set origin
        """
        return [getattr(point, "altitude") for point in self.data_points]

    def get_relative_times(self) -> list[int]:
        """
        Gets a chronological list of all the relative time stamps of GPS points
        from their set origin
        """
        return [getattr(point, "relative_time") for point in self.data_points]


class GpsPoint:
    """
    Stores a single GPS log outputted by a GPS device
    """

    def __init__(self, unix_time, lat, lon, alt, time_offset, start_northing, start_easting, start_altitude):
        self.relative_time = unix_time - time_offset
        self.altitude = alt
        self.easting, self.northing = (int(ls[0]) for ls in convert_bng(lon, lat))

        self.displacement = math.sqrt(
            (self.northing - start_northing) ** 2
            + (self.easting - start_easting) ** 2
            + (self.altitude - start_altitude) ** 2
        )

    def set_displacement_from_point(self, start_easting, start_northing, start_altitude):
        self.displacement = math.sqrt(
            (self.northing - start_northing) ** 2
            + (self.easting - start_easting) ** 2
            + (self.altitude - start_altitude) ** 2
        )

    def __str__(self):
        return (f"GpsData | Time: {self.relative_time:.3f}, Northing: {self.northing}, Easting: {self.easting}, "
                f"Alt: {self.altitude}, Displacement: {self.displacement:.3f}")