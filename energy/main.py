# Standard Library Imports
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from itertools import product


# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
# Set a professional style for the plot
plt.style.use('_mpl-gallery')
import pandas as pd
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore")

# Local Imports
from utils_general import *
from utils_energy import *
from preprocessing import *
from metrics import *

# Definimos paths que son necesarios. 
xls_events = 'Eventos_24hrs.xlsx' # Catálogo original de eventos
events_over = 'sismos_txt/times_events_over_four.txt' # catálogo sobre 4 dado por Aaron
inventory_path = "inventory" # para quitar la respuesta del instrumento
stations_names = ['CO10','AC04', 'AC05', 'CO05']

file_over = pd.read_csv(events_over)
file_under = pd.read_excel(xls_events)

# Station coordinates
stations_coord = {
    'CO10': (-29.24, -71.46),
    'AC04': (-28.20, -71.07),
    'AC05': (-28.84, -70.27),
    'CO05': (-29.92, -71.24)
    
}

sample_rate = 40
intervals = 1273 # intervalos de tiempo que voy a tomar, i.e. trazas de tiempo de ruido que voy a tomar

# P-wave propagation speed
v_P = 8.064

# ------------------------------------
# Preprocessing
# ------------------------------------
path_folder_traces = "señales_sismos/BHZ"
# Se quita la respuesta instrumental
files_bhz = find_files(path_folder_traces, ['.mseed'])
files_bhz_ch, key_names_bhz = sort_files(files_bhz, '.mseed')
processed_stations = {}

for station in key_names_bhz:
    processed_stations[station] = process_station(files_bhz_ch, station, inventory_path)


st_AC04_BHZ = processed_stations[key_names_bhz[0]].select(channel='BHZ')
st_AC05_BHZ = processed_stations[key_names_bhz[1]].select(channel='BHZ')
st_CO05_BHZ  = processed_stations[key_names_bhz[2]].select(channel='BHZ')
st_CO10_BHZ = processed_stations[key_names_bhz[3]].select(channel='BHZ')


# Crear un diccionario para mapear los nombres de las estaciones a las trazas de las estaciones
stations_dic = {name: globals()[f'st_{name}_BHZ'][0] for name in stations_names}


def main(station: str, magnitudes: np.array, method: str):

    if station == "first":
        station_selected = 'CO10'
        st_selected = 0

    elif station == "second":
        station_selected = 'AC04'
        st_selected = 1
        
    no_event_df = pd.read_csv(f'sismos_txt/no_event_intervals_{station_selected}.txt')

    best_magnitude, best_labels, best_data, opt_thr, power_events = find_best_magnitude(file_over = file_over, file_under = file_under, no_event_df = no_event_df,
                                                                                        stations_coord = stations_coord, stations_names = stations_names, stations_dic_tr = stations_dic, 
                                                                                        magnitudes = magnitudes, method = method, st_selection = st_selected)
    return best_magnitude, best_labels, best_data, opt_thr, power_events, station_selected


def main2(station: str, magnitude: int, method: str):

    if station == "first":
        station_selected = 'CO10'
        st_selected = 0

    elif station == "second":
        station_selected = 'AC04'
        st_selected = 1

    no_event_df = pd.read_csv(f'sismos_txt/no_event_intervals_{station_selected}.txt')

    data, optminal_thrs, labels, events = test_magnitude(file_over = file_over, file_under = file_under, no_event_df = no_event_df,
                                                                           stations_coord = stations_coord, stations_names = stations_names, stations_dic_tr = stations_dic,
                                                                           magnitude = magnitude, method = method, station = st_selected)
    return data, optminal_thrs, labels, events, station_selected

if __name__ == '__main__':

    st = input("Choose the station. If you want the closest type 'first', if you want the second closest type 'second':")
    methods = ['youden_index', 'euclidean_distance', 'concordance_probability']
    method_to_test = methods[1]

    #magnitudes_a_probar = np.arange(3.0, 5.6, 0.1).round(1)
    #magnitude, labels, data, optminal_thrs, events, station = main(st, magnitudes_a_probar, method_to_test)

    
    magnitude = float(input("Enter the magnitude to separate the events: "))
    data, optminal_thrs, labels, events, station = main2(st, magnitude, method_to_test)
    #station = station_no_event[0]
    #Acá abajo elegir 1 si se quiere analizar para una magnitud en particular o 0 si se quiere ver eventos vs no eventos
    #optminal_thrs = optminal_thrs[1]
    #labels = labels[1]

    event_type = [f'$M \geq {magnitude}$', f'$M < {magnitude}$', 'Ruido (sin eventos)']
    classes_1 = ['No Event', 'Event']
    classes_2 = [f'No M>={magnitude}', f'M>={magnitude}']


    # ------------------------------------
    # Plot of all the things
    # ------------------------------------

    plot_power(events, station=station, n_frames=10, use_log=True, event_type=event_type)

    plot_roc_curve(labels, station, data, class_type=classes_2, method = method_to_test, title='ROC Curve for power')
    plot_confusion_matrix2(optminal_thrs, station, method_to_test, labels, data, classes_2)

    plt.show()
