from geopy.distance import geodesic
from obspy.signal.trigger import classic_sta_lta, trigger_onset, plot_trigger
from obspy import read
from obspy import Trace
import numpy as np
import math

def inicializar_sta_lta(traza, nsta, nlta):
    """
    Inicializa el cálculo STA/LTA para los primeros 30 segundos de la traza.   
    """
    cft = classic_sta_lta(traza.data, int(nsta), int(nlta))
    return cft

def actualizar_sta_lta(cft_anterior, traza_anterior, traza_nueva, nsta, nlta):
    """
    Actualiza el cálculo STA/LTA añadiendo 10 segundos nuevos y eliminando los 10 segundos más antiguos.
    """
    # Concatenar los últimos datos de la traza anterior con la traza nueva
    traza_actualizada = np.concatenate((traza_anterior.data[-nlta:], traza_nueva.data))

    # Calcular STA/LTA para la traza actualizada
    cft_actualizado = classic_sta_lta(traza_actualizada, int(nsta), int(nlta))

    # Devolver los últimos valores del cálculo STA/LTA, que corresponden a la ventana actualizada
    return cft_actualizado[-len(traza_nueva):]


def p_picking_val(stations, ventana_10s, ventana_30s, nsta, nlta, v_P, coord_list, thr_on, thr_off):

    """
    Función que realiza el picking de la onda P. En caso de que se detecte un sismo en las estación principal,
    se verifica que las estaciones adyacentes también lo detecten.

    Entradas:
    ----------
    st_main: obspy.core.stream.Stream
        Stream de la estación principal.
    adj_st_1: obspy.core.stream.Stream
        Stream de la estación adyacente 1.
    adj_st_2: obspy.core.stream.Stream
        Stream de la estación adyacente 2.
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
        cft_main = actualizar_sta_lta(cft_main, initial_tr_main , new_tr_main, int(nsta * fs), int(nlta * fs))

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
                cft_adj_1 = actualizar_sta_lta(cft_adj_1, initial_tr_adj_1, new_tr_adj_1, int(nsta * fs), int(nlta * fs))
                cft_adj_2 = actualizar_sta_lta(cft_adj_2, initial_tr_adj_2, new_tr_adj_2, int(nsta * fs), int(nlta * fs))
                cft_adj_3 = actualizar_sta_lta(cft_adj_3, initial_tr_adj_3, new_tr_adj_3, int(nsta * fs), int(nlta * fs))

                if np.any(cft_adj_1 > thr_on) and np.any(cft_adj_2 > thr_on) and np.any(cft_adj_3 > thr_on):
                    time_all_str = new_tr_main.stats.starttime.strftime("%Y-%m-%dT%H:%M")
                    time_trigger_all_comp = [t.strftime("%Y-%m-%dT%H:%M") for t in time_trigger_all]
                    if time_all_str not in time_trigger_all_comp:
                        time_trigger_all.append(new_tr_main.stats.starttime)
    

                initial_tr_adj_1 = new_tr_adj_1
                initial_tr_adj_2 = new_tr_adj_2


        initial_tr_main  = new_tr_main
        
    return time_trigger_main, time_trigger_all


def p_picking_each(station, ventana_10s, ventana_30s, nsta, nlta, thr_on, thr_off):

    """
    Función que realiza el picking de la onda P para solo una estación durante la traza completa y guarda los tiempos en que 
    hay trigger.

    Entradas:
    ----------
    station: obspy.core.stream.Stream
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
    tr_main = station.select(channel='BHZ')[0]

    # Frecuencia de muestreo
    fs = tr_main.stats.sampling_rate

    # Convertir los tamaños de ventana a muestras
    muestras_10s = int(ventana_10s * fs)
    muestras_30s = int(ventana_30s * fs)

    # Inicializar el cálculo STA/LTA para los primeros 30 segundos de la traza.
    #cft_main = inicializar_sta_lta(tr_main, int(nsta * fs), int(nlta * fs))
    cft_main = classic_sta_lta(tr_main.data, int(nsta * fs), int(nlta * fs))
    initial_tr_main = tr_main.slice(endtime=tr_main.stats.starttime + ventana_30s)

    time_trigger = []

    for i in range(10, len(tr_main), muestras_10s):
        # Tomar la sección actual de 30 segundos
        end_window = i + muestras_30s
        if end_window > len(tr_main):
            break
        new_tr_main = tr_main.slice(starttime = tr_main.stats.starttime + i/fs, endtime = tr_main.stats.starttime + end_window/fs)
        #cft_main = actualizar_sta_lta(initial_tr_main , new_tr_main, int(nsta * fs), int(nlta * fs))
        cft_main = classic_sta_lta(new_tr_main, int(nsta * fs), int(nlta * fs))

        # Si se activa se plotea
        if np.any(cft_main > thr_on):
            # Find the time index where the condition is met
            trigger_time_index = np.argmin(cft_main > thr_on)
            # Convert the index to seconds
            trigger_time_seconds = i + trigger_time_index


            # Convertimos starttime a una cadena que solo contiene la fecha, la hora y los minutos
            time_main_str = new_tr_main.stats.starttime.strftime("%Y-%m-%dT%H:%M")
            # Creamos una lista separada para las comparaciones
            time_trigger_main_comp = [t.strftime("%Y-%m-%dT%H:%M") for t in time_trigger]

            #durante los próximos 30 segundos buscamos el peak de la amplitud de la señal

            
            # No aseguramos de que el tiempo de inicio del sismo no se haya registrado antes
            if time_main_str not in time_trigger_main_comp:
                time_trigger.append(new_tr_main.stats.starttime + trigger_time_seconds / fs)


        initial_tr_main  = new_tr_main
    return time_trigger