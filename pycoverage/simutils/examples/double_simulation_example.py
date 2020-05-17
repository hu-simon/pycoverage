""" A basic, low-fidelity double simulation that uses the BasicSimulation class from the MissionControl module. 

This is a low-fidelity simulation to test the coverage control tracking algorithm.

"""

import time

import numpy as np
from pycoverage.vorutils import pyvoro
from pycoverage.simutils.MissionControl import BasicSimulation
from pycoverage.simutils.MissionControl import BasicDoubleSimulation
