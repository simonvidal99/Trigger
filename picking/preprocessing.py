# Standard Library Imports
import datetime

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
# Set a professional style for the plot
plt.style.use('_mpl-gallery')
from obspy import read, UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset, plot_trigger
from obspy import Trace
from obspy.imaging.spectrogram import spectrogram

import pandas as pd

# Local Imports
from .p_picking import p_picking_all, p_picking_each, p_picking_val
from .utils_energy import *
from .utils_general import *
from .plot import *


def process_station(files_bhz_ch, station_name, inventory_path):
    st_files = files_bhz_ch[station_name]
    st_raw = read(st_files[0])
    st_raw += read(st_files[1])
    st_raw += read(st_files[2])

    remove_file = os.path.join(inventory_path, f"C1_{station_name.split('/')[-1]}.xml")

    st_resp = st_raw.copy()
    st_removed = remove_response(st_resp.select(channel='BHZ')[0], remove_file , 'obspy')
    st_resp[2] = st_removed

    assert(st_resp.select(channel='BHZ')[0] == st_removed)

    st = st_resp.copy()
    st.filter('bandpass', freqmin=4.0, freqmax=10.0)

    return st




