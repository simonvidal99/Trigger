# Standard Library Imports
import os
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from collections import OrderedDict
from itertools import product
from itertools import groupby
import itertools
import pickle
import glob
import chardet
import copy
from natsort import natsorted

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
# Set a professional style for the plot
plt.style.use('_mpl-gallery')
import matplotlib.dates as mdaates
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from obspy import read, UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset, plot_trigger
from obspy import Trace, Stream
from obspy.imaging.spectrogram import spectrogram
import pandas as pd
from geopy.distance import geodesic
from tqdm.auto import tqdm
from icecream import ic
import seaborn as sns
from scipy.stats import norm, kstest, skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

# Local Imports
from energy.utils_general import *
from energy.utils_energy import *
from energy.preprocessing import *
from energy.metrics import *
from picking.p_picking import p_picking_each




def z_channel_dict(folders_signals: list, inventory_path: str):

    result_dict_z = OrderedDict()
    #result_dict_e = OrderedDict()
    #result_dict_n = OrderedDict()


    for folder in folders_signals:
        waveform_type = os.path.split(folder)[-1]
        events_id = natsorted(glob.glob(os.path.join(folder, '*')))

        for event_id in events_id:
            events_name = os.path.split(event_id)[-1]
            network_path = glob.glob(os.path.join(event_id, '*'))

            for network_folder in network_path:
                network_name = os.path.split(network_folder)[-1]
                
                stations_path = glob.glob(os.path.join(network_folder, '*'))

                for station_path in stations_path:
                    stations_names = os.path.split(station_path)[-1]
                    #ic(stations_names)
                    stations_path = glob.glob(os.path.join(station_path, '*'))
                    #ic(stations_path)

                    stations_ch_z = [station_z for station_z in stations_path if 'BHZ' in station_z]
                    stations_ch_e = [station_e for station_e in stations_path if 'BHE' in station_e]
                    stations_ch_n = [station_n for station_n in stations_path if 'BHN' in station_n]
                    
                    traces = []

                    #ic(len(stations_ch_z))
                    # Read and store traces using ObsPy
                    # traces_z = []

                    # Files to remove response
                    file_response = glob.glob(os.path.join(inventory_path, f"*{stations_names}.xml"))
                    # #ic(file_response)


                    # trace_z = read(stations_ch_z[0])[0]  # Assuming one trace per file
                    
                    if file_response:
                        remove_file_z = file_response[0]

                    # #ic(remove_file_z)
                    # trace_resp_z = trace_z.copy()
                    # trace_removed_z = remove_response(trace_resp_z, remove_file_z , 'obspy')
                    # #st_resp[2] = st_removed
                    # st_z = trace_removed_z.copy()
                    # st_z.filter('bandpass', freqmin=4.0, freqmax=10.0)
                    
                    # traces_z.append(st_z)



                    trace_Z = read(stations_ch_z[0])[0]
                    trace_resp = trace_Z.copy()
                    trace_resp = remove_response(trace_resp, remove_file_z , 'obspy')
                    trace = Stream(traces = trace_resp)
                    if len(stations_ch_e) > 0:
                        trace += read(stations_ch_e[0])[0]
                    if len(stations_ch_n) > 0:    
                        trace += read(stations_ch_n[0])[0]
                    

                    # if file_response:
                    #     remove_file = file_response[0]

                    # ic(remove_file)

                    # tr_resp = trace.copy()
                    # for i in range(len(trace)):
                    #     st_removed = remove_response(tr_resp.select(channel=tr_resp[i].stats.channel)[0], remove_file , 'obspy')
                    #     tr_resp[i] = st_removed
                    #     ic(i)

                    tr_filtered = trace.copy()
                    tr_filtered.filter('bandpass', freqmin=4.0, freqmax=10.0)
                    traces.append(tr_filtered)

                    result_dict_z.setdefault(waveform_type, {}).setdefault(events_name, {}).setdefault(network_name, {}).setdefault(stations_names, traces)


    return result_dict_z




def raw_signal_dict(folders_signals: list, inventory_path: str):

    no_filter = OrderedDict()
    #result_dict_e = OrderedDict()
    #result_dict_n = OrderedDict()


    for folder in folders_signals:
        waveform_type = os.path.split(folder)[-1]
        events_id = natsorted(glob.glob(os.path.join(folder, '*')))

        for event_id in events_id:
            events_name = os.path.split(event_id)[-1]
            network_path = glob.glob(os.path.join(event_id, '*'))

            for network_folder in network_path:
                network_name = os.path.split(network_folder)[-1]
                
                stations_path = glob.glob(os.path.join(network_folder, '*'))

                for station_path in stations_path:
                    stations_names = os.path.split(station_path)[-1]
                    #ic(stations_names)
                    stations_path = glob.glob(os.path.join(station_path, '*'))
                    #ic(stations_path)

                    stations_ch_z = [station_z for station_z in stations_path if 'BHZ' in station_z]
                    stations_ch_e = [station_e for station_e in stations_path if 'BHE' in station_e]
                    stations_ch_n = [station_n for station_n in stations_path if 'BHN' in station_n]
                    
                    traces = []

                    #ic(len(stations_ch_z))
                    # Read and store traces using ObsPy
                    # traces_z = []

                    # Files to remove response
                    file_response = glob.glob(os.path.join(inventory_path, f"*{stations_names}.xml"))
                    # #ic(file_response)


                    # trace_z = read(stations_ch_z[0])[0]  # Assuming one trace per file
                    
                    if file_response:
                        remove_file_z = file_response[0]

                    # #ic(remove_file_z)
                    # trace_resp_z = trace_z.copy()
                    # trace_removed_z = remove_response(trace_resp_z, remove_file_z , 'obspy')
                    # #st_resp[2] = st_removed
                    # st_z = trace_removed_z.copy()
                    # st_z.filter('bandpass', freqmin=4.0, freqmax=10.0)
                    
                    # traces_z.append(st_z)



                    trace_Z = read(stations_ch_z[0])[0]
                    trace_resp = trace_Z.copy()
                    #trace_resp = remove_response(trace_resp, remove_file_z , 'obspy')
                    trace = Stream(traces = trace_resp)
                    if len(stations_ch_e) > 0:
                        trace += read(stations_ch_e[0])[0]
                    if len(stations_ch_n) > 0:    
                        trace += read(stations_ch_n[0])[0]
                    

                    # if file_response:
                    #     remove_file = file_response[0]

                    # ic(remove_file)

                    # tr_resp = trace.copy()
                    # for i in range(len(trace)):
                    #     st_removed = remove_response(tr_resp.select(channel=tr_resp[i].stats.channel)[0], remove_file , 'obspy')
                    #     tr_resp[i] = st_removed
                    #     ic(i)

                    tr_filtered = trace.copy()
                    #tr_filtered.filter('bandpass', freqmin=4.0, freqmax=10.0)
                    traces.append(tr_filtered)

                    no_filter.setdefault(waveform_type, {}).setdefault(events_name, {}).setdefault(network_name, {}).setdefault(stations_names, traces)


    return no_filter


def filter_csv(files_raw, events_ids):
    # Columnas a mantener
    columns_to_keep = ["#EventID", "Time", "Latitude", "Longitude", "Magnitude", "Estacion"]

    for i, (waveform_type, event_ids) in enumerate(events_ids.items()):
        # Leer el archivo CSV
        df = pd.read_csv(files_raw[i], sep=';')

        # Filtrar las filas donde '#EventID' está en event_ids
        df = df[df['#EventID'].isin(event_ids)]

        # Ordernar la columna de #EventID
        df = df.sort_values(by='#EventID')

        # Mantener solo las columnas deseadas
        df = df[columns_to_keep]

        # Guardar el nuevo DataFrame en un nuevo archivo CSV
        df.to_excel(os.path.join("BD paper", "catalogos", f"{waveform_type}_filtered.xlsx"), index=False)



def closest_station(file_path, stations_names, stations_coord_all, v_P, n_closest_stations = 7):
    '''
    Actualiza todos los archivos csv con la estación más cercana a cada evento sísmico.
    '''

    for file in file_path:
        ic(file)

        df_events = pd.read_excel(file)

        # Cambiar nombre de la columna "Time" a "Fecha UTC" y cosas en inglés por español
        df_events = df_events.rename(columns={'Time': 'Fecha UTC', 'Latitude': 'Latitud', 'Longitude': 'Longitud', 'Magnitude': 'Magnitud'})
        
        # Cambiar esta columna a formato UTC
        df_events['Fecha UTC'] = pd.to_datetime(df_events['Fecha UTC'])
        #df_events = df_events.rename(columns={'Latitude': 'Latitud', 'Longitude': 'Longitud'})

        df_events.to_excel(file, index=False)

        # Tomar el valor máximo y mínimo de la magnitud en el dataframe
        max_magnitude = df_events['Magnitud'].max()
        min_magnitude = df_events['Magnitud'].min()

        df_new = calculate_detection_times(df_events, stations_coord_all, v_P, magnitude_range=(min_magnitude, max_magnitude))
        _, closest_sts_names = nearest_n_stations(df_new, stations_names, n_closest_stations)

        df_events = pd.read_excel(file)

        # Agregar columnas para cada estación cercana
        for i in range(n_closest_stations):
            col_name = f'Estación más cercana {i+1}'
            df_events[col_name] = closest_sts_names[i]

        df_events.to_excel(file, index=False)

        df_events = pd.read_excel(file)

        # Agregar columnas para el tiempo de inicio de cada estación cercana
        for i in range(n_closest_stations):
            col_name = f'Inicio estación más cercana {i+1}'
            df_events[col_name] = df_events.apply(lambda row: df_new.loc[row.name, f'Inicio_{row[f"Estación más cercana {i+1}"]}'], axis=1)

        # Escribir en el archivo excel
        df_events.to_excel(file, index=False)




def find_matching_trace_v2(event_dict, dataframe, waveform_key, station_columns:list):
    trace_dict = {}  # Cambié trace_list por trace_dict

    event_dict_waveform = event_dict[waveform_key]

    for i in range(len(dataframe['#EventID'].astype(str))):
        station_data = {key: value for sublist in
                        list(event_dict_waveform[dataframe['#EventID'].astype(str).iloc[i]].values()) for key, value
                        in sublist.items()}
        station_names = list(station_data.keys())

        for j, column in enumerate(dataframe.loc[:, station_columns]):
            station_column_value = dataframe[column].astype(str).iloc[i]
            matching_station = next((station for station in station_names if station in station_column_value), None)

            if matching_station is not None:
                start_time_column = f"Inicio estación más cercana {j + 1}"
                start_time = UTCDateTime(dataframe[start_time_column].iloc[i])

                trace = station_data[matching_station][0]
                trace[0].stats.starttime = start_time

                # Cambiado trace_list por trace_dict
                if dataframe['#EventID'].iloc[i] not in trace_dict:
                    trace_dict[dataframe['#EventID'].iloc[i]] = []
                trace_dict[dataframe['#EventID'].iloc[i]].extend(station_data[matching_station])

                break

    return trace_dict