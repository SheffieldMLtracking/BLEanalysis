from convertbng.util import convert_bng

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

    def __str__(self):
        return (f"Transmitter ID: {self.tx_id}; lon, lat: ({self.longitude}, {self.latitude}); altitude: {self.altitude}"
                f"easting, northing: ({self.easting}, {self.northing})")