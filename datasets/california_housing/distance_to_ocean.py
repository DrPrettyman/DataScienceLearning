"""
Define a function to calculate the distance of a location in california to the 
nearest coast.

Also create a sklearn transformer to add this distance to a numpy array given the 
column indices of the longitude and latitude.
"""

import numpy as np
from shapely.geometry import LineString, Point
from pyproj import Geod

# Define the california coastline
_ca_coast = [
        (42.0, -124.2),  # CA-OR Border
        (41.7, -124.2),
        (40.8, -124.2),  # Eureka
        (39.3, -123.8),
        (38.3, -123.1),  # Point Reyes
        (37.8, -122.5),  # San Francisco
        (36.6, -122.0),  # Monterey
        (35.6, -121.2),  # Big Sur
        (34.5, -120.5),  # Santa Barbara
        (34.0, -118.5),  # Los Angeles
        (32.7, -117.2),  # San Diego
        (32.5, -117.1)   # CA-Mexico Border
    ]
COASTLINE = LineString([(lon, lat) for lat, lon in _ca_coast])


def distance_to_coast(lon: float, lat: float,) -> float:
    """
    Calculate the distance from a point to the nearest ocean coastline.
    
    Parameters:
    lon (float): Longitude of the point
    lat (float): Latitude of the point
    
    Returns:
    float: Distance to nearest ocean in kilometers
    """

    point = Point(lon, lat)
    
    # Find nearest point on coastline
    nearest_point = COASTLINE.interpolate(COASTLINE.project(point))
    
    # Calculate geodesic distance
    geod = Geod(ellps='WGS84')
    distance = geod.inv(lon, lat, 
                       nearest_point.x, 
                       nearest_point.y)[2]
    
    # Convert to kilometers
    distance_km = distance / 1000
    
    # Determine if point is east or west of coastline
    # Simplified check: if longitude is less than nearest coast point, we're in ocean: return 0
    return 0 if lon < nearest_point.x else distance_km
