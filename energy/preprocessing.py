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
from metrics import *

# Cuando corra el enery jupyter debe estar así:
# from .utils_energy import *
# from .utils_general import *
# from .metrics import *


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


def data_power(file_over: pd.DataFrame, file_under: pd.DataFrame, no_event_df: pd.DataFrame, stations_coord: dict, stations_names: list, stations_dic: dict, 
               magnitude: float, st_selection: int, v_P: float = 8.046, sample_rate: int = 40, pre_event: int = 0, intervals = 1273):

    # ------------------------------------
    # Separing the data using the magnitude given in the input 
    # ------------------------------------

    df_under = calculate_detection_times(file_under, stations_coord, v_P, magnitude_range = (0, magnitude-0.1))
    df_over = file_over[file_over['Magnitud'] >= magnitude]

    # ------------------------------------
    # Events M>=4 and M<4 station selection
    # ------------------------------------
    start_times_over_, closest_sts_names_over_  = nearest_two_stations(df_over, stations_names)
    
    # M>magnitude
    closest_sts_tr_over = [stations_dic[estacion] for estacion in closest_sts_names_over_[st_selection]] 
    start_times_over , closest_sts_names_over  = start_times_over_[st_selection], closest_sts_names_over_[st_selection]

    # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
    start_traces = [sts.slice(start) for sts, start in zip(closest_sts_tr_over, start_times_over)]

    # Se calcula el punto donde cada traza tendría su finalización del evento
    end_events_traces = [endpoint_event(st.data)[st_selection] for st in start_traces]

    post_event = end_events_traces
    sliced_traces = [traces.slice(start - pre_event, start + post_event[i]*sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr_over , start_times_over))]


    _, power_events_over = zip(*[energy_power(st.data) for st in sliced_traces])


    # M < magnitude
    start_times_under_, closest_sts_under_ = nearest_two_stations(df_under, stations_names)

    start_times_under , closest_sts_names_under = start_times_under_[st_selection], closest_sts_under_[st_selection] 

    closest_sts_tr_under = [stations_dic[estacion] for estacion in closest_sts_names_under] 

    # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
    start_tr_under = [sts.slice(start) for sts, start in zip(closest_sts_tr_under, start_times_under)]

    # Se calcula el punto donde cada traza tendría su finalización del evento
    end_events_tr_under = [endpoint_event(st.data)[st_selection] for st in start_tr_under]

    post_event_under = end_events_tr_under
    sample_rate = 40
    sliced_traces_under_four = [traces.slice(start - pre_event, start + post_event_under[i]*sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr_under, start_times_under))]

    _, power_events_under = zip(*[energy_power(st.data) for st in sliced_traces_under_four])

    
    # NO eventos (RUIDO)
    if st_selection == 0:
        st = closest_sts_under_[0][0]
    elif st_selection == 1:
        st = closest_sts_under_[1][0]
    
    random_intervals = no_event_df.sample(n=intervals, random_state=1)

    # Convertir las columnas 'Start' y 'End' a datetime
    random_intervals['Start'] = pd.to_datetime(random_intervals['Start'])
    random_intervals['End'] = pd.to_datetime(random_intervals['End'])

    # Crear las listas start_times_no_events y end_time_no_events
    start_times_no_events = random_intervals['Start'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z').tolist()
    end_time_no_events = random_intervals['End'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z').tolist()

    start_times_no_events = [UTCDateTime(time) for time in start_times_no_events]
    end_time_no_events = [UTCDateTime(time) for time in end_time_no_events]
    # start_times_no_events.sort()
    # end_time_no_events.sort()

    station_no_event = [st]*intervals
    closest_sts_tr_no_event = [stations_dic[estacion] for estacion in station_no_event]

    sliced_traces_no_event = [traces.slice(start, start + 60) for traces, start in zip(closest_sts_tr_no_event, start_times_no_events)]

    _, power_events_no_event = zip(*[energy_power(st.data) for st in sliced_traces_no_event])

    power_events_all = [power_events_over, power_events_under, power_events_no_event]

    power_last_frame = [arr[-1] for tup in power_events_all for arr in tup]
    data_power = np.array(power_last_frame)
    #ic(data_power)
    # Como estamos trabajando con el log de 10 antes, lo hacemos tambien acá, hace todo más bonito jjjjjeee
    log_data_power = np.log10(data_power)

    return log_data_power, power_events_all



def label_power(power_events_all: np.array):
    
    labels_power = np.concatenate([np.ones(len(power_events_all[0])),
                            np.zeros(len(power_events_all[1])),
                            np.zeros(len(power_events_all[2]))])

    return labels_power



def find_best_magnitude(file_over: pd.DataFrame, file_under: pd.DataFrame, no_event_df: pd.DataFrame ,stations_coord: dict , stations_names: list, stations_dic_tr: dict,
                         magnitudes: np.array, method: str, st_selection: int, v_P: float = 8.046, sample_rate: int = 40, pre_event:int = 0):

    fp = float('inf')
    fn = float('inf')

    for i, magnitude in enumerate(magnitudes):

        log_data_power, power_events_all = data_power(file_over = file_over, file_under = file_under, no_event_df = no_event_df, stations_coord = stations_coord , stations_names = stations_names, 
                                                      stations_dic = stations_dic_tr, magnitude = magnitude, st_selection = st_selection, v_P = 8.046, sample_rate = 40, pre_event = 0)

        labels_power = label_power(power_events_all)

        #fpr, tpr, thr = roc_curve(labels_power, log_data_power)


        opt_thr = calculate_optimal_threshold(labels_power, log_data_power, method = method)
        predicted_labels = np.array([1 if x >= opt_thr else 0 for x in log_data_power])
        f_p = confusion_matrix(labels_power, predicted_labels)[0][1]
        f_n = confusion_matrix(labels_power, predicted_labels)[1][0]
    
        if f_p < fp and f_n <= fn:
            best_magnitude = magnitude
            #ic("Mejoró la magnitud", best_magnitude)
            best_labels = labels_power
            best_data = log_data_power
            opt_thr = opt_thr
            power_events = power_events_all
            fp = f_p
            fn = f_n
            #ic("Nuevos mejores parámetros", fp,fn)
        
    #     if f_p == 0 and f_n == 0:
    #         ic(f_p, f_n)
    #         return best_magnitude, best_labels, best_data, opt_thr
    # cz = tpr * (1 - fpr)
    # cz = np.argmax(cz)
    # if cz > best_thr:
    #     best_magnitude = magnitude
    #     best_labels = labels_power
    #     best_data = log_data_power
    #     opt_thr = thr[cz]
    #     best_thr = cz

    # Calcular la distancia de Euclides entre (0,1) y el punto en la curva ROC
    # distance = np.sqrt(1e8*(0 - fpr)**2 + 1e8*(1 - tpr)**2)
    # distance_value = np.min(distance)
    # distance_index = np.argmin(distance)
    # distance_value = distance_value
    # Actualizar la magnitud si encontramos un valor con menor distancia
    # if distance_value < best_distance:
    #     best_distance = distance_value
    #     best_magnitude = magnitude
    #     best_labels = labels_power
    #     best_data = log_data_power
    #     opt_thr = thr[distance_index]

    return best_magnitude, best_labels, best_data, opt_thr, power_events




def test_magnitude(file_over: pd.DataFrame, file_under: pd.DataFrame, no_event_df: pd.DataFrame, stations_coord: dict, stations_names: list, stations_dic_tr: dict, 
            magnitude: float, method: str, station: str, v_P: float = 8.046, sample_rate: int = 40, intervals = 1273):

    # ------------------------------------
    # Separing the data using the magnitude given in the input
    # ------------------------------------
    

    # Calculate detection times and format DataFrame
    #df_over = calculate_detection_times(df, stations_coord, v_P, magnitude_range = (magnitude,10)) Esto cuando no tenga el catálogo de Aaron
    df_over = file_over[file_over['Magnitud'] >= magnitude]
    df_under = calculate_detection_times(file_under, stations_coord, v_P, magnitude_range = (0,magnitude-0.1))


    # ------------------------------------
    # Events M>=4 and M<4 station selection
    # ------------------------------------

    # Getting beggining and end of the events for the 2 closest stations 
    #start_time_over, closest_st_names_over = nearest_station(file_path, stations_names)
    start_times_over, closest_sts_names_over  = nearest_two_stations(df_over, stations_names)
    start_times_under, closest_sts_under = nearest_two_stations(df_under, stations_names)

    # Here we choose wether we are going to work with the closest or with the second closest
    if station == "first":
        start_time_over, closest_st_names_over = start_times_over[0], closest_sts_names_over[0]
        start_time_under, closest_st_under = start_times_under[0], closest_sts_under[0]

    elif station == "second":
        start_time_over, closest_st_names_over = start_times_over[1], closest_sts_names_over[1]
        start_time_under, closest_st_under = start_times_under[1], closest_sts_under[1]

    station_no_event = closest_st_names_over
    ic(station_no_event)
    # ------------------------------------
    # Events M>=magnitude processing
    # ------------------------------------

    # Crear una lista con las estaciones más cercanas para cada evento
    closest_sts_tr = [stations_dic_tr[estacion] for estacion in closest_st_names_over]

    # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
    start_traces = [sts.slice(start) for sts, start in zip(closest_sts_tr, start_time_over)]
    # Se calcula el punto donde cada traza tendría su finalización del evento
    end_events_traces = [endpoint_event(st.data)[1] for st in start_traces]

    post_event = end_events_traces
    sliced_traces = [traces.slice(start, start + post_event[i]*sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr, start_time_over))]

    # Calculate energy and power 
    _, power_events = zip(*[energy_power(st.data) for st in sliced_traces])


    # ------------------------------------
    # Events M<4 processing
    # ------------------------------------

    #start_time_under, closest_sts_under = nearest_station(events_under, stations_names)
    closest_sts_tr_under = [stations_dic_tr[estacion] for estacion in closest_st_under]

    # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
    start_tr_under = [sts.slice(start) for sts, start in zip(closest_sts_tr_under, start_time_under)]
    # Se calcula el punto donde cada traza tendría su finalización del evento
    end_events_tr_under = [endpoint_event(st.data)[1] for st in start_tr_under]

    post_event_under = end_events_tr_under
    sliced_traces_under = [traces.slice(start , start + post_event_under [i]*sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr_under, start_time_under))]

    # Calculate energy and power 
    _, power_events_under = zip(*[energy_power(st.data) for st in sliced_traces_under])

    # ------------------------------------
    # No Events (Noise)
    # ------------------------------------
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
    closest_sts_tr_no_event = [stations_dic_tr[estacion] for estacion in station_no_event]

    sliced_traces_no_event = [traces.slice(start, start + 60) for traces, start in zip(closest_sts_tr_no_event, start_times_no_events)]

    # Calculate energy and power 
    energy_events_no_events, power_events_no_event = zip(*[energy_power(st.data) for st in sliced_traces_no_event])

    # ------------------------------------
    # Adding all of them to plot histogram
    # ------------------------------------

    power_events_all = [power_events, power_events_under, power_events_no_event]

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

    data = data_power
    optminal_thrs = [optimal_thr_power_class1, optimal_thr_power_class2]
    labels = [labels_power_class1, labels_power_class2]
    events = power_events_all

    return data, optminal_thrs, labels, events, station_no_event