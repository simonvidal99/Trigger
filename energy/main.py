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


# Este script es una copia en .py del jupyter energy.ipynb
file_path = "catalog_new_events.txt" # catalogo de eventos sobre 4
inventory_path = "inventory"
stations_names = ['CO10','AC04', 'AC05', 'CO05']
pre_event = 0
sample_rate = 40

events_under_four = "times_events_under_four.txt" # catalogo de eventos bajo 4

#no_event_file = 'no_event_intervals.txt' # df de tiempos de trazas sin eventos
intervals = 1273 # intervalos de tiempo que voy a tomar, i.e. trazas de tiempo de ruido que voy a tomar

#station_no_event = ['CO10']

event_type = ['$M \geq 4$', '$M < 4$', 'Ruido (sin eventos)']

classes_1 = ['No Event', 'Event']
classes_2 = ['No M>=4', 'M>=4']



# ------------------------------------
# Preprocessing
# ------------------------------------
files_bhz = find_files("señales_sismos/BHZ", ['.mseed'])
files_bhz_ch, key_names_bhz = sort_files(files_bhz, '.mseed')
processed_stations = {}

for station in key_names_bhz:
    processed_stations[station] = process_station(files_bhz_ch, station, inventory_path)


st_AC04 = processed_stations[key_names_bhz[0]]
st_AC05 = processed_stations[key_names_bhz[1]]
st_CO05 = processed_stations[key_names_bhz[2]]
st_CO10 = processed_stations[key_names_bhz[3]]

st_AC04_BHZ = st_AC04.select(channel='BHZ')
st_AC05_BHZ = st_AC05.select(channel='BHZ')
st_CO05_BHZ = st_CO05.select(channel='BHZ')
st_CO10_BHZ = st_CO10.select(channel='BHZ')

# Crear un diccionario para mapear los nombres de las estaciones a las estaciones
stations_dic = {name: globals()[f'st_{name}_BHZ'][0] for name in stations_names}


def main(station: str):

    # ------------------------------------
    # Events M>=4 and M<4 station selection
    # ------------------------------------

    # Gettin beggining and end of the events for the 2 closest stations 
    #start_time_over_four, closest_st_names_over_four = nearest_station(file_path, stations_names)
    start_times_over_four, closest_sts_names_over_four  = nearest_two_stations(file_path, stations_names)
    start_times_under_four, closest_sts_under_four = nearest_two_stations(events_under_four, stations_names)

    # Here we choose wether we are going to work with the closest or with the second closest
    if station == "first":
        start_time_over_four, closest_st_names_over_four = start_times_over_four[0], closest_sts_names_over_four[0]
        start_time_under_four, closest_st_under_four = start_times_under_four[0], closest_sts_under_four[0]
        station_no_event = ['CO10']
        no_event_file = 'no_event_intervals_CO10.txt'

    elif station == "second":
        start_time_over_four, closest_st_names_over_four = start_times_over_four[1], closest_sts_names_over_four[1]
        start_time_under_four, closest_st_under_four = start_times_under_four[1], closest_sts_under_four[1]
        station_no_event = ['AC04']
        no_event_file = 'no_event_intervals_AC04.txt'

    # ------------------------------------
    # Events M>=4 processing
    # ------------------------------------

    # Crear una lista con las estaciones más cercanas para cada evento
    closest_sts_tr = [stations_dic[estacion] for estacion in closest_st_names_over_four]

    # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
    start_traces = [sts.slice(start) for sts, start in zip(closest_sts_tr, start_time_over_four)]
    # Se calcula el punto donde cada traza tendría su finalización del evento
    end_events_traces = [endpoint_event(st.data)[1] for st in start_traces]

    post_event = end_events_traces
    sliced_traces = [traces.slice(start - pre_event, start + post_event[i]/sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr, start_time_over_four))]

    # Calculate energy and power 
    energy_events, power_events = zip(*[energy_power(st.data) for st in sliced_traces])


    # ------------------------------------
    # Events M<4 processing
    # ------------------------------------

    #start_time_under_four, closest_sts_under_four = nearest_station(events_under_four, stations_names)
    closest_sts_tr_under_four = [stations_dic[estacion] for estacion in closest_st_under_four]

    # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
    start_tr_under_four = [sts.slice(start) for sts, start in zip(closest_sts_tr_under_four, start_time_under_four)]
    # Se calcula el punto donde cada traza tendría su finalización del evento
    end_events_tr_under_four = [endpoint_event(st.data)[1] for st in start_tr_under_four]

    post_event_under_four = end_events_tr_under_four
    sliced_traces_under_four = [traces.slice(start - pre_event, start + post_event_under_four [i]/sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr_under_four, start_time_under_four))]

    # Calculate energy and power 
    energy_events_under_four, power_events_under_four = zip(*[energy_power(st.data) for st in sliced_traces_under_four])

    # ------------------------------------
    # No Events (Noise)
    # ------------------------------------

    no_event_df = pd.read_csv(no_event_file)
    random_intervals = no_event_df.sample(n=intervals, random_state=1)

    # Convertir las columnas 'Start' y 'End' a datetime
    random_intervals['Start'] = pd.to_datetime(random_intervals['Start'])
    random_intervals['End'] = pd.to_datetime(random_intervals['End'])

    # Crear las listas start_times_no_events y end_time_no_events
    start_times_no_events = random_intervals['Start'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z').tolist()
    end_time_no_events = random_intervals['End'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z').tolist()

    start_times_no_events = [UTCDateTime(time) for time in start_times_no_events]
    end_time_no_events = [UTCDateTime(time) for time in end_time_no_events]
    start_times_no_events.sort()
    end_time_no_events.sort()

    station_no_event = station_no_event*intervals
    closest_sts_tr_no_event = [stations_dic[estacion] for estacion in station_no_event]

    sliced_traces_no_event = [traces.slice(start, start + 60) for traces, start in zip(closest_sts_tr_no_event, start_times_no_events)]

    # Calculate energy and power 
    energy_events_no_events, power_events_no_event = zip(*[energy_power(st.data) for st in sliced_traces_no_event])

    # ------------------------------------
    # Adding all of them to plot histogram
    # ------------------------------------

    power_events_all = [power_events, power_events_under_four, power_events_no_event]
    energy_events_all = [energy_events, energy_events_under_four, energy_events_no_events]

    # ------------------------------------
    # ROC curve anc Confussion Matrix with Energy as criteria
    # ------------------------------------

    energy_events_flattened = [np.concatenate(events) for events in energy_events_all]
    data_energ = np.concatenate(energy_events_flattened)
    # Como estamos trabajando con el log de 10 antes, lo hacemos tambien acá, hace todo más bonito jjjjjeee
    data_energ = np.log10(data_energ)
    labels_energy_class1 = np.concatenate([np.ones(len(energy_events_flattened[0])),
                            np.ones(len(energy_events_flattened[1])),
                            np.zeros(len(energy_events_flattened[2]))])
    
    optimal_thr_energy_class1 = calculate_optimal_threshold(labels_energy_class1, data_energ)


    labels_energy_class2  = np.concatenate([np.ones(len(energy_events_flattened[0])),
                         np.zeros(len(energy_events_flattened[1])),
                         np.zeros(len(energy_events_flattened[2]))])
    
    optimal_thr_energy_class2 = calculate_optimal_threshold(labels_energy_class2, data_energ)


    
    # ------------------------------------
    # ROC curve anc Confussion Matrix with Power as criteria
    # ------------------------------------

    # La siguiente linea de código toma la potencia en el último frame para cada evento (o no evento)
    power_last_frame = [arr[-1] for tup in power_events_all for arr in tup]
    data_power = np.array(power_last_frame)
    # Como estamos trabajando con el log de 10 antes, lo hacemos tambien acá, hace todo más bonito jjjjjeee
    data_power = np.log10(data_power)

    labels_power_class1 = np.concatenate([np.ones(len(power_events_all[0])),
                            np.ones(len(power_events_all[1])),
                            np.zeros(len(power_events_all[2]))])
    
    optimal_thr_power_class1 = calculate_optimal_threshold(labels_power_class1, data_power)

    labels_power_class2 = np.concatenate([np.ones(len(power_events_all[0])),
                                np.zeros(len(power_events_all[1])),
                                np.zeros(len(power_events_all[2]))])
    
    optimal_thr_power_class2 = calculate_optimal_threshold(labels_power_class2, data_power)

    data = [data_energ, data_power]
    optminal_thrs = [optimal_thr_energy_class1, optimal_thr_energy_class2, optimal_thr_power_class1, optimal_thr_power_class2]
    labels = [labels_energy_class1, labels_energy_class2, labels_power_class1, labels_power_class2]
    events = [energy_events_all, power_events_all]

    return data, optminal_thrs, labels, events, station_no_event



if __name__=='__main__':
    input = input("Choose the station. If you want the closest type 'first', if you want the second closest type 'second':")

    data, optminal_thrs, labels, events, station = main(input)
    station = station[0]

    # ------------------------------------
    # Plot of all the things
    # ------------------------------------

    plot_energy_hist(events[0], station = station, frame = 3, use_log = True, event_type = event_type)
    plot_power(events[1], station = station, n_frames=10, use_log=True, event_type=event_type)

    plot_roc_curve(labels[0], station, data[0])
    plot_confusion_matrix(optminal_thrs[0], station, 'youden_index', labels[0], data[0], classes_1)

    plot_roc_curve(labels[1], station, data[0])
    plot_confusion_matrix(optminal_thrs[1], station, 'youden_index', labels[1], data[0], classes_2)

    plot_roc_curve(labels[2], station, data[1])
    plot_confusion_matrix(optminal_thrs[2], station, 'youden_index', labels[2], data[1], classes_1)

    plot_roc_curve(labels[3], station, data[1])
    plot_confusion_matrix(optminal_thrs[3], station, 'youden_index', labels[3], data[1], classes_2)


    