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
# Se quita la respuesta instrumental
files_bhz = find_files("señales_sismos/BHZ", ['.mseed'])
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


def main2(station: str, magnitudes: list):
        
    best_distance = float('inf')
    for magnitude in magnitudes:
        # ------------------------------------
        # Separing the data using the magnitude given in the input
        # ------------------------------------

        # Read Excel file
        file_over = pd.read_csv(events_over)
        file_under = pd.read_excel(xls_events)
        

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
            station_no_event = ['CO10']
            no_event_file = 'sismos_txt/no_event_intervals_CO10.txt'

        elif station == "second":
            start_time_over, closest_st_names_over = start_times_over[1], closest_sts_names_over[1]
            start_time_under, closest_st_under = start_times_under[1], closest_sts_under[1]
            station_no_event = ['AC04']
            no_event_file = 'sismos_txt/no_event_intervals_AC04.txt'

        # ------------------------------------
        # Events M>=magnitude processing
        # ------------------------------------

        # Crear una lista con las estaciones más cercanas para cada evento
        closest_sts_tr = [stations_dic[estacion] for estacion in closest_st_names_over]

        # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
        start_traces = [sts.slice(start) for sts, start in zip(closest_sts_tr, start_time_over)]
        # Se calcula el punto donde cada traza tendría su finalización del evento
        end_events_traces = [endpoint_event(st.data)[1] for st in start_traces]

        post_event = end_events_traces
        sliced_traces = [traces.slice(start, start + post_event[i]*sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr, start_time_over))]

        # Calculate energy and power 
        energy_events, power_events = zip(*[energy_power(st.data) for st in sliced_traces])


        # ------------------------------------
        # Events M<4 processing
        # ------------------------------------

        #start_time_under, closest_sts_under = nearest_station(events_under, stations_names)
        closest_sts_tr_under = [stations_dic[estacion] for estacion in closest_st_under]

        # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
        start_tr_under = [sts.slice(start) for sts, start in zip(closest_sts_tr_under, start_time_under)]
        # Se calcula el punto donde cada traza tendría su finalización del evento
        end_events_tr_under = [endpoint_event(st.data)[1] for st in start_tr_under]

        post_event_under = end_events_tr_under
        sliced_traces_under = [traces.slice(start , start + post_event_under [i]*sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr_under, start_time_under))]

        # Calculate energy and power 
        energy_events_under, power_events_under = zip(*[energy_power(st.data) for st in sliced_traces_under])

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

        power_events_all = [power_events, power_events_under, power_events_no_event]
        energy_events_all = [energy_events, energy_events_under, energy_events_no_events]

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


        fpr, tpr, thr = roc_curve(labels_power_class2, data_power)

        # Calcular la distancia de Euclides entre (0,1) y el punto en la curva ROC
        distance = np.sqrt((0 - fpr)**2 + (1 - tpr)**2)
        
        distance = np.min(distance)
        distance_value = np.min(distance)
        #distance_index = np.argmin(distance)

        # Actualizar la magnitud si encontramos un valor con menor distancia
        if distance_value < best_distance:
            best_distance = distance_value
            best_magnitude = magnitude
            #best_labels = labels_power_class2
            #best_data = data_power
            #opt_thr = thr[distance_index]

            data = [data_energ, data_power]
            optminal_thrs = [optimal_thr_energy_class1, optimal_thr_energy_class2, optimal_thr_power_class1, optimal_thr_power_class2]
            labels = [labels_energy_class1, labels_energy_class2, labels_power_class1, labels_power_class2]
            events = [energy_events_all, power_events_all]

    return data, optminal_thrs, labels, events, station_no_event, best_magnitude



if __name__ == '__main__':

    st = input("Choose the station. If you want the closest type 'first', if you want the second closest type 'second':")
    #magnitude = float(input("Enter the magnitude to separate the events: "))
    magnitudes_a_probar = np.linspace(3.0,5.5,21)

    data, optminal_thrs, labels, events, station, magnitude = main2(st, magnitudes_a_probar)
    event_type = [f'$M \geq {magnitude}$', f'$M < {magnitude}$', 'Ruido (sin eventos)']
    classes_1 = ['No Event', 'Event']
    classes_2 = [f'No M>={magnitude}', f'M>={magnitude}']
    station = station[0]

    # ------------------------------------
    # Plot of all the things
    # ------------------------------------

    method = ['youden_index', 'euclidean_distance', 'concordance_probability']

    plot_power(events[1], station=station, n_frames=10, use_log=True, event_type=event_type)

    #plot_roc_curve(labels[2], station, data[1], class_type=classes_1, title='ROC Curve for power')
    #plot_confusion_matrix2(optminal_thrs[2], station, method[2], labels[2], data[1], classes_1)

    plot_roc_curve(labels[3], station, data[1], class_type=classes_2, method = method[1], title='ROC Curve for power')
    plot_confusion_matrix2(optminal_thrs[3], station, method[1], labels[3], data[1], classes_2)
