from geopy.distance import geodesic
from obspy.signal.trigger import classic_sta_lta, trigger_onset, plot_trigger
from obspy import read
from obspy import Trace
import numpy as np
import math
from tqdm.auto import tqdm

from icecream import ic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor




def inicializar_sta_lta(traza, nsta, nlta):
    """
    Inicializa el cálculo STA/LTA para los primeros 30 segundos de la traza.
    """
    cft = classic_sta_lta(traza.data, int(nsta), int(nlta))
    return cft

def actualizar_sta_lta(traza_anterior, traza_nueva, nsta, nlta):
    """
    Actualiza el cálculo STA/LTA añadiendo 10 segundos nuevos y eliminando los 10 segundos más antiguos.
    """
    # Concatenar los últimos datos de la traza anterior con la traza nueva
    traza_actualizada = np.concatenate((traza_anterior.data[-nlta:], traza_nueva.data))

    # Calcular STA/LTA para la traza actualizada
    cft_actualizado = classic_sta_lta(traza_actualizada, int(nsta), int(nlta))

    # Devolver los últimos valores del cálculo STA/LTA, que corresponden a la ventana actualizada
    return cft_actualizado[-len(traza_nueva):]



def p_picking_all(stations, ventana_10s, ventana_30s, nsta, nlta, v_P, coord_list, thr_on, thr_off):

    """
    Función que realiza el picking de la onda P. En caso de que se detecte un sismo en las estación principal,
    se verifica que las estaciones adyacentes también lo detecten.


    Entradas:
    ----------
    stations: list
        Lista de las estaciones en formato obspy.core.stream.Stream.
    ventana_10s: int
        Tiempo de la ventana en segundos donde se aplica el STA/LTA.
    ventana_30s: int
        Tiempo de la señal en segundos que se analiza en cada iteración.
    nsta: int
        Largo del short time average en segundos.
    nlta: int
        Largo del long time average en segundos.
    v_P: float
        Velocidad de propagación de la onda P en km/s.
    coords_st: list
        Coordenadas (latitud, longitud) de las estaciones.
    thr_on: float
        Umbral de activación del picking.
    thr_off: float
        Umbral de desactivación del picking.
    
    Salidas:
    ----------
    time_trigger_all: list
        Lista que contiene tiempos cuando todas las estaciones detectan un evento
    """
    # Coordenadas (latitud, longitud) de las estaciones y selección de canal para cada estación
    station_coords = [coord for coord in coord_list]
    station_traces = [station.select(channel='BHZ')[0] for station in stations]

    # Distancia y tiempo de llegada de la onda P entre todas las estaciones
    distance_matrix = [[geodesic(coord1, coord2).kilometers for coord2 in station_coords] for coord1 in station_coords]
    time_matrix = [[d_val / v_P for d_val in d_row] for d_row in distance_matrix]

    # Frecuencia de muestreo
    fs = station_traces[0].stats.sampling_rate

    # Convertir los tamaños de ventana a muestras
    muestras_10s = int(ventana_10s * fs)
    muestras_30s = int(ventana_30s * fs)

    # Inicializar el cálculo STA/LTA para los primeros 30 segundos de todas las trazas.
    cft_all = [inicializar_sta_lta(tr, int(nsta * fs), int(nlta * fs)) for tr in station_traces]
    initial_tr_all = [tr.slice(endtime=tr.stats.starttime + ventana_30s) for tr in station_traces]

    #time_trigger_main = []
    time_trigger_all = []


    with ThreadPoolExecutor() as executor:
        for i in range(10, len(station_traces[0]), muestras_10s):
            # Tomar la sección actual de 30 segundos
            end_window = i + muestras_30s
            if end_window > len(station_traces[0]):
                break
            # Actualizar el cálculo STA/LTA para todas las trazas
            new_tr_all = [tr.slice(starttime=tr.stats.starttime + i / fs, endtime=tr.stats.starttime + end_window / fs) for tr in station_traces]

            # Crear una lista de tuplas de argumentos para la función actualizar_sta_lta
            args_list = [(initial_tr.data, new_tr.data, int(nsta * fs), int(nlta * fs)) for initial_tr, new_tr in zip(initial_tr_all, new_tr_all)]
            cft_all = list(executor.map(actualizar_sta_lta, *zip(*args_list)))

            # Verificar si alguna estación activa la detección
            triggered_stations = [i for i, cft in enumerate(cft_all) if np.any(cft > thr_on)]

            if triggered_stations:
                #ic(triggered_stations)
                # Tomar la primera estación que activa la detección
                triggered_station = triggered_stations[0]

                # Tiempo de la estación de referencia a todas las demás estaciones
                times_to_station = time_matrix[triggered_station]
                # índices de las dos estaciones con los tiempos de llegada más cortos (excluyendo la propia estación de referencia)
                closest_station_idx = sorted(range(len(times_to_station)), key=lambda i: times_to_station[i])[1:3]
                # Tiempos de llegada a las dos estaciones más cercanas
                times_to_closest_stations = sorted(times_to_station)[1:3]
                # Trazas de las dos estaciones más cercanas
                closest_station_traces = [station_traces[i] for i in closest_station_idx]

                # se inicializan las trazas de las estaciones aledañas
                initial_tr_adjs =  [tr.slice(starttime = new_tr_all[triggered_station].stats.starttime,
                                             endtime=new_tr_all[triggered_station].stats.endtime) for tr in closest_station_traces]
                
                for j in range(math.ceil(max(times_to_closest_stations)/10)+1):
                    # Slice de las estaciones aledañas para el cálculo STA/LTA posterior
                    new_tr_adjs = [tr.slice(starttime=new_tr_all[triggered_station].stats.starttime + j * ventana_10s, 
                                        endtime=new_tr_all[triggered_station].stats.endtime + j * ventana_10s) 
                                        for tr in closest_station_traces]
                    
                    # Cálculo STA/LTA para las estaciones aledañas
                    args_list_adj = [(initial_tr, new_tr, int(nsta * fs), int(nlta * fs)) for initial_tr, new_tr in zip(initial_tr_adjs, new_tr_adjs)]
                    cft_adjs = list(executor.map(actualizar_sta_lta, *zip(*args_list_adj)))


                    # Verificar si alguna de las estaciones aledañas detecta el evento y guarda el tiempo cuando fue detectada por primera vez
                    triggered_stations_adjs = [i for i, cft in enumerate(cft_adjs) if all(np.any(array > thr_on) for array in cft_adjs)]
                    if len(triggered_stations_adjs) == 2:
                        #ic(triggered_stations_adjs)
                        time_all_str = new_tr_all[triggered_station].stats.starttime.strftime("%Y-%m-%dT%H:%M")
                        time_trigger_all_comp = [t.strftime("%Y-%m-%dT%H:%M") for t in time_trigger_all]
                        if time_all_str not in time_trigger_all_comp:
                            time_trigger_all.append(new_tr_all[triggered_station].stats.starttime)
        
                    initial_tr_adjs = new_tr_adjs

            initial_tr_all  = new_tr_all
            
    return time_trigger_all





def p_picking_val(stations, ventana_10s, ventana_30s, nsta, nlta, v_P, coord_list, thr_on, thr_off):

    """
    Función que realiza el picking de la onda P para una estación arbitraria como principal. En caso de que exista trigger,
    se verifica que las estaciones adyacentes también lo detecten.

    Entradas:
    ----------
    stations: list
        Lista de las estaciones en formato obspy.core.stream.Stream.
    ventana_10s: int
        Tiempo de la ventana en segundos donde se aplica el STA/LTA.
    ventana_30s: int
        Tiempo de la señal en segundos que se analiza en cada iteración.
    nsta: int
        Largo del short time average en segundos.
    nlta: int
        Largo del long time average en segundos.
    v_P: float
        Velocidad de propagación de la onda P en km/s.
    coords_st: list
        Coordenadas (latitud, longitud) de las estaciones.
    thr_on: float
        Umbral de activación del picking.
    thr_off: float
        Umbral de desactivación del picking.
    
    Salidas:
    ----------
    time_trigger_main: list
        Lista que contiene tiempos cuando solo la estación principal detecta un evento
    time_trigger_all: list
        Lista que contiene tiempos cuando todas las estaciones detectan un evento
    """
    # Estaciones
    st_main = stations[0]
    adj_st_1 = stations[1]
    adj_st_2 = stations[2]
    adj_st_3 = stations[3]


    # Coordenadas (latitud, longitud) de las estaciones
    coord_main = coord_list[0]
    coord_adj_1 = coord_list[1]
    coord_adj_2 = coord_list[2]
    coord_adj_3 = coord_list[3]

    # Distancia entre las estaciones
    d_main_adj1 = geodesic(coord_main, coord_adj_1).kilometers
    d_main_adj2 = geodesic(coord_main, coord_adj_2 ).kilometers
    d_main_adj3 = geodesic(coord_main , coord_adj_3 ).kilometers

    # Tiempo de llegada de la onda P a las estaciones
    t_P_main_adj1 = d_main_adj1 / v_P
    t_P_main_adj2 = d_main_adj2 / v_P
    t_P_main_adj3 = d_main_adj3 / v_P

    # Elección de canal para cada estación. 
    tr_main = st_main.select(channel='BHZ')[0]
    tr_adj_1 = adj_st_1.select(channel='BHZ')[0]
    tr_adj_2 = adj_st_2.select(channel='BHZ')[0]
    tr_adj_3 = adj_st_3.select(channel='BHZ')[0]

    # Frecuencia de muestreo
    fs = tr_main.stats.sampling_rate

    # Convertir los tamaños de ventana a muestras
    muestras_10s = int(ventana_10s * fs)
    muestras_30s = int(ventana_30s * fs)

    # Inicializar el cálculo STA/LTA para los primeros 30 segundos de la traza.
    cft_main = inicializar_sta_lta(tr_main, int(nsta * fs), int(nlta * fs))
    initial_tr_main = tr_main.slice(endtime=tr_main.stats.starttime + ventana_30s)

    time_trigger_main = []
    time_trigger_all = []

    for i in range(10, len(tr_main), muestras_10s):
        # Tomar la sección actual de 30 segundos
        end_window = i + muestras_30s
        if end_window > len(tr_main):
            break
        new_tr_main = tr_main.slice(starttime = tr_main.stats.starttime + i/fs, endtime = tr_main.stats.starttime + end_window/fs)
        cft_main = actualizar_sta_lta(initial_tr_main , new_tr_main, int(nsta * fs), int(nlta * fs))

        # Si se activa se plotea
        if np.any(cft_main > thr_on):
            # Convertimos starttime a una cadena que solo contiene la fecha, la hora y los minutos
            time_main_str = new_tr_main.stats.starttime.strftime("%Y-%m-%dT%H:%M")
            # Creamos una lista separada para las comparaciones
            time_trigger_main_comp = [t.strftime("%Y-%m-%dT%H:%M") for t in time_trigger_main]
            
            # No aseguramos de que el tiempo de inicio del sismo no se haya registrado antes
            if time_main_str not in time_trigger_main_comp:
                time_trigger_main.append(new_tr_main.stats.starttime)

            # Verificar si las estaciones adyacentes también detectan un sismo en ese intervalo de tiempo
            initial_tr_adj_1 = tr_adj_1.slice(starttime = new_tr_main.stats.starttime, endtime = new_tr_main.stats.endtime)
            initial_tr_adj_2 = tr_adj_2.slice(starttime = new_tr_main.stats.starttime, endtime = new_tr_main.stats.endtime)
            initial_tr_adj_3 = tr_adj_3.slice(starttime = new_tr_main.stats.starttime, endtime = new_tr_main.stats.endtime)
            cft_adj_1 = inicializar_sta_lta(initial_tr_adj_1, int(nsta * fs), int(nlta * fs))
            cft_adj_2 = inicializar_sta_lta(initial_tr_adj_2, int(nsta * fs), int(nlta * fs))
            cft_adj_3 = inicializar_sta_lta(initial_tr_adj_3, int(nsta * fs), int(nlta * fs))
            
            for j in range(math.ceil(max(t_P_main_adj1, t_P_main_adj2, t_P_main_adj3)/10)+1):
                new_tr_adj_1 = tr_adj_1.slice(starttime = new_tr_main.stats.starttime + j*ventana_10s, 
                                              endtime = new_tr_main.stats.endtime + j*ventana_10s)
                new_tr_adj_2 = tr_adj_2.slice(starttime = new_tr_main.stats.starttime + j*ventana_10s, 
                                              endtime = new_tr_main.stats.endtime + j*ventana_10s)
                new_tr_adj_3 = tr_adj_3.slice(starttime = new_tr_main.stats.starttime + j*ventana_10s, 
                                              endtime = new_tr_main.stats.endtime + j*ventana_10s)
                cft_adj_1 = actualizar_sta_lta(initial_tr_adj_1, new_tr_adj_1, int(nsta * fs), int(nlta * fs))
                cft_adj_2 = actualizar_sta_lta(initial_tr_adj_2, new_tr_adj_2, int(nsta * fs), int(nlta * fs))
                cft_adj_3 = actualizar_sta_lta(initial_tr_adj_3, new_tr_adj_3, int(nsta * fs), int(nlta * fs))

                if np.any(cft_adj_1 > thr_on) and np.any(cft_adj_2 > thr_on) and np.any(cft_adj_3 > thr_on):
                    time_all_str = new_tr_main.stats.starttime.strftime("%Y-%m-%dT%H:%M")
                    time_trigger_all_comp = [t.strftime("%Y-%m-%dT%H:%M") for t in time_trigger_all]
                    if time_all_str not in time_trigger_all_comp:
                        time_trigger_all.append(new_tr_main.stats.starttime)
    

                initial_tr_adj_1 = new_tr_adj_1
                initial_tr_adj_2 = new_tr_adj_2


        initial_tr_main  = new_tr_main
        
    return time_trigger_main, time_trigger_all



def p_picking_each(stations, ventana_10s, ventana_30s, nsta, nlta, thr_on, thr_off):

    """
    Función que realiza el picking de la onda P para solo una estación durante la traza completa y guarda los tiempos en que 
    hay trigger.

    Entradas:
    ----------
    stations: obspy.core.stream.Stream
        Estación a analizar.
    ventana_10s: int
        Tiempo de desplazamiento de la ventana en segundos.
    ventana_30s: int
        Tiempo de la señal en segundos donde se aplica el STA/LTA.
    nsta: int
        Largo del short time average en segundos.
    nlta: int
        Largo del long time average en segundos.
    thr_on: float
        Umbral de activación del picking.
    thr_off: float
        Umbral de desactivación del picking.
    
    Salidas:
    ----------
    time_trigger_all: list
        Lista que contiene tiempos cuando todas las estaciones detectan un evento
    """

    # Elección de canal para la estación
    tr_main = stations.select(channel='BHZ')[0]

    # Frecuencia de muestreo
    fs = tr_main.stats.sampling_rate

    # Convertir los tamaños de ventana a muestras
    muestras_10s = int(ventana_10s * fs)
    muestras_30s = int(ventana_30s * fs)

    # Inicializar el cálculo STA/LTA para los primeros 30 segundos de la traza.
    cft_main = inicializar_sta_lta(tr_main, int(nsta * fs), int(nlta * fs))
    initial_tr_main = tr_main.slice(endtime=tr_main.stats.starttime + ventana_30s)

    time_trigger = []

    for i in range(10, len(tr_main), muestras_10s):
        # Tomar la sección actual de 30 segundos
        end_window = i + muestras_30s
        if end_window > len(tr_main):
            break
        new_tr_main = tr_main.slice(starttime = tr_main.stats.starttime + i/fs, endtime = tr_main.stats.starttime + end_window/fs)
        cft_main = actualizar_sta_lta(initial_tr_main , new_tr_main, int(nsta * fs), int(nlta * fs))

        # Si se activa se plotea
        if np.any(cft_main > thr_on):
            # Convertimos starttime a una cadena que solo contiene la fecha, la hora y los minutos
            time_main_str = new_tr_main.stats.starttime.strftime("%Y-%m-%dT%H:%M")
            # Creamos una lista separada para las comparaciones
            time_trigger_main_comp = [t.strftime("%Y-%m-%dT%H:%M") for t in time_trigger]
            
            # No aseguramos de que el tiempo de inicio del sismo no se haya registrado antes
            if time_main_str not in time_trigger_main_comp:
                time_trigger.append(new_tr_main.stats.starttime)


        initial_tr_main  = new_tr_main
    return time_trigger
