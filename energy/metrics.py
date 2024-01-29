# Standard Library Imports
import datetime

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from obspy import read, UTCDateTime

# Cuando corra el main debe estar así
# from utils_energy import *
# from utils_general import *

# Cuando corra el enery jupyter debe estar así:
from .utils_energy import *
from .utils_general import *



def plot_roc_curve(labels, station, data, class_type, method ,title='ROC Curve'):
    # Ajusta el tamaño de la figura
    plt.figure(figsize=(10, 6))

    fpr, tpr, _ = roc_curve(labels, data)
    roc_auc = auc(fpr, tpr)

    # Calcula el umbral óptimo
    optimal_threshold = calculate_optimal_threshold(labels, data, method)

    #ax.figure(figsize=(9, 5))

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)

    # linea diagonal
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{title}. Station {station}. Classes {class_type}. Optimal Threshold: {optimal_threshold:.2f}', fontsize=14)    
    # grid
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(loc="lower right", fontsize=10)

    # grid más detallada
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.3)

    # Show the plot
    #plt.show()



def calculate_optimal_threshold(labels, data, method='youden_index'):
    fpr, tpr, thresholds = roc_curve(labels, data)

    if method == 'youden_index':
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    elif method == 'euclidean_distance':
        distances = np.sqrt((0 - fpr)**2 + (1 - tpr)**2)
        index = np.argmin(distances)
        optimal_threshold = thresholds[index]
    elif method == 'concordance_probability':
        cz = tpr * (1 - fpr)
        optimal_threshold = thresholds[np.argmax(cz)]
    else:
        raise ValueError("Invalid method")

    return optimal_threshold


def find_best_magnitude(file_over,file_under, stations_coord ,stations_names, stations_dic, magnitudes, st_selection = 0, v_P = 8.046, sample_rate = 40, pre_event = 0):
    best_magnitude = None
    best_distance = float('inf')

    for magnitude in magnitudes:

        df_under = calculate_detection_times(file_under, stations_coord, v_P, magnitude_range = (0, magnitude-0.1))

        df_over = file_over[file_over['Magnitud'] >= magnitude]
        start_times_over_, closest_sts_names_over_  = nearest_two_stations(df_over, stations_names)
        start_times_over , closest_sts_names_over  = start_times_over_[st_selection], closest_sts_names_over_[st_selection]

        # Acá se elige si se quiere la traza de la estación más cercana (i = 0) o de la segunda más cercana (i = 1)
        st_selection = st_selection

        # M>magnitude
        closest_sts_tr_over = [stations_dic[estacion] for estacion in closest_sts_names_over] 

        # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
        start_traces = [sts.slice(start) for sts, start in zip(closest_sts_tr_over, start_times_over)]

        # Se calcula el punto donde cada traza tendría su finalización del evento
        end_events_traces = [endpoint_event(st.data)[st_selection] for st in start_traces]

        post_event = end_events_traces
        sliced_traces = [traces.slice(start - pre_event, start + post_event[i]*sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr_over , start_times_over))]


        _, power_events_over = zip(*[energy_power(st.data) for st in sliced_traces])


        # M < magnitude
        start_times_under, closest_sts_under = nearest_two_stations(df_under, stations_names)

        start_times_under , closest_sts_names_under = start_times_under[st_selection], closest_sts_under[st_selection] 

        closest_sts_tr_under= [stations_dic[estacion] for estacion in closest_sts_names_under] 

        # Tomamos trazas que parten en el inicio de cada evento y toman todo el resto de la señal 
        start_tr_under_four = [sts.slice(start) for sts, start in zip(closest_sts_tr_under, start_times_under)]

        # Se calcula el punto donde cada traza tendría su finalización del evento
        end_events_tr_under_four = [endpoint_event(st.data)[st_selection] for st in start_tr_under_four]

        pre_event_under_four = 0
        post_event_under_four = end_events_tr_under_four
        sample_rate = 40
        sliced_traces_under_four = [traces.slice(start - pre_event_under_four, start + post_event_under_four [i]*sample_rate) for i, (traces, start) in enumerate(zip(closest_sts_tr_under, start_times_under))]

        _, power_events_under = zip(*[energy_power(st.data) for st in sliced_traces_under_four])

        if st_selection == 0:
            st = 'CO10'
        elif st_selection == 1:
            st = 'AC04'
        
        # NO eventos
        no_event_df = pd.read_csv('sismos_txt/no_event_intervals_AC04.txt')

        intervals = 1273
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

        station_no_event = [st]*intervals
        closest_sts_tr_no_event = [stations_dic[estacion] for estacion in station_no_event]

        sliced_traces_no_event = [traces.slice(start, start + 60) for traces, start in zip(closest_sts_tr_no_event, start_times_no_events)]

        _, power_events_no_event = zip(*[energy_power(st.data) for st in sliced_traces_no_event])

        power_events_all = [power_events_over, power_events_under, power_events_no_event]

        power_last_frame = [arr[-1] for tup in power_events_all for arr in tup]
        data_power = np.array(power_last_frame)
        # Como estamos trabajando con el log de 10 antes, lo hacemos tambien acá, hace todo más bonito jjjjjeee
        log_data_power = np.log10(data_power)

        labels_power = np.concatenate([np.ones(len(power_events_all[0])),
                                    np.zeros(len(power_events_all[1])),
                                    np.zeros(len(power_events_all[2]))])
        fpr, tpr, thr = roc_curve(labels_power, log_data_power)

        # Calcular la distancia de Euclides entre (0,1) y el punto en la curva ROC
        distance = np.sqrt(1e8*(0 - fpr)**2 + 1e8*(1 - tpr)**2)
        distance_value = np.min(distance)
        distance_index = np.argmin(distance)

        distance_value = distance_value

        # Actualizar la magnitud si encontramos un valor con menor distancia
        if distance_value < best_distance:
            best_distance = distance_value
            best_magnitude = magnitude
            best_labels = labels_power
            best_data = log_data_power
            opt_thr = thr[distance_index]

    return best_magnitude, best_distance/1e4, best_labels, best_data, opt_thr




