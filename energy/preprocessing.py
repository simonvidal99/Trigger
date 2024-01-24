# Standard Library Imports
import datetime

# Third-Party Library Imports
import numpy as np
import os
from collections import defaultdict
from itertools import product
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
# Set a professional style for the plot
plt.style.use('_mpl-gallery')
from obspy import read, UTCDateTime
from obspy import Trace
import sys
sys.path.insert(0, 'Trigger/energy')

# Local Imports
# HABLAR CON CAMILO PORQUE NO SE COMO ARREGLAR ESTE PROBLEMA
# Cuando corra el main debe estar así
from utils_energy import *
from utils_general import *

# Cuando corra el enery jupyter debe estar así:
# from .utils_energy import *
# from .utils_general import *


def find_files(path, extensions):
    '''
    Entrega de vuelta un diccionario con la extensión del archivo como llave, y el path a cada archivo como valor

    '''
    file_list = {ext: [] for ext in list(extensions)}
    for root, dirs, files in os.walk(path):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    file_list[ext].append(os.path.join(root, file))

    return file_list


def sort_files(file_dic, extension):
    '''
    Junta los archivos por carpeta y canal. Queda un diccionario que tiene como llave la carpeta y el canal, 
    y como valor una lista con los paths a los archivos de esa carpeta y canal.   
    '''
    # Diccionario para almacenar los archivos por carpeta y canal
    grouped_files = defaultdict(list)
    key_names = []
    
    # Ordenar los archivos alfabéticamente antes de procesarlos
    sorted_files = sorted(file_dic[extension])
    
    for file in sorted_files:
        # Extraer la carpeta y el canal del path del archivo
        parts = os.path.split(file)
        folder = parts[0]  # La carpeta es el primer elemento en el path
        channel = parts[1].split('_')[0]  # El canal es el primer elemento en el nombre del archivo
        key = os.path.join(folder, channel)
        if [key] not in key_names:
            key_names.append([key])
        # Agrupar los archivos
        grouped_files[key].append(file)

    key_names = list(itertools.chain(*key_names))
    return grouped_files, key_names


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






