"""
data_path.py -- where the INSANE data set is.

Set the environment variable INSANE_DATA_DIR to the folder that holds the flight folders
(indoor_1_sensors, mars_5_sensors, ...), or put that folder at data/9DOF inside the repository.
The folder itself must be called 9DOF (data_loader.py looks for it by name).
"""
import os

DATA_DIR = os.environ.get('INSANE_DATA_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '9DOF')
