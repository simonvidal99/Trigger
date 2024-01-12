import pandas as pd

from geopy.distance import geodesic
import numpy as np

from datetime import datetime
from datetime import timedelta

from tqdm.auto import tqdm
from icecream import ic
from .p_picking import p_picking_all, p_picking_val, p_picking_each
import csv
from itertools import product
import os
import pickle

from scipy.stats import norm

from concurrent.futures import ThreadPoolExecutor, as_completed




def calculate_detection_times(df, stations_coord, v_P, magnitude_thr=3.5):
    """
    Calcula los tiempos de detección de los eventos sísmicos reales en las distintas estaciones y guarda los resultados en un DataFrame.
    Esto sería tener un DataFrame con el tiempo real de los eventos y el tiempo cuando deberían ser detectados por cada estación.

    Enradas:
    - df: DataFrame con los datos de los eventos sísmicos.
    - stations_coord: Diccionario con las coordenadas de las estaciones.
    - v_P: Velocidad de la onda P.
    - magnitude_thr: Umbral de magnitud para filtrar los eventos sísmicos.
    
    Salida:
    - formatted_df: DataFrame con los tiempos de detección de los eventos sísmicos en todas las estaciones y la magnitud del evento
    """
    for station, coords in stations_coord.items():
        df[f'Hora detección estación {station}'] = df.apply(
            lambda row: row['Fecha UTC'] + timedelta(seconds=geodesic((row['Latitud'], row['Longitud']), coords).kilometers / v_P),
            axis=1
        )
        df[f'Distancia a estación {station}'] = df.apply(
            lambda row: round(geodesic((row['Latitud'], row['Longitud']), coords).kilometers, 2),
            axis=1
        )


    formatted_df = df.copy()
    
    # Formatea las columnas de tiempo
    for col in formatted_df.columns:
        if 'Fecha UTC' in col or 'Hora detección estación' in col:
            formatted_df[col] = formatted_df[col].apply(lambda time: time.strftime('%Y-%m-%dT%H:%M:%S') if pd.notnull(time) else '')

    # Selecciona solo las columnas de tiempo, magnitud y distancia
    columns = ['Fecha UTC', 'Magnitud'] + [f'Hora detección estación {station}' for station in stations_coord.keys()] + [f'Distancia a estación {station}' for station in stations_coord.keys()]
    formatted_df = formatted_df[columns]
    
    # Filtra eventos con magnitud mayor o igual a un umbral 
    formatted_df = formatted_df[formatted_df['Magnitud'] >= magnitude_thr]

    return formatted_df




def load_best_params(best_params, filename):
    """
    Carga los mejores parámetros desde un archivo o los guarda en un archivo si no existe.

    Entradas:
    - best_params: Diccionario con los mejores parámetros.
    - filename: Nombre del archivo donde se guardan los mejores parámetros.

    Salida:
    - best_params: Diccionario con los mejores parámetros.
    """

    if os.path.exists(filename):
        # Cargar best_params desde un archivo
        with open(filename, 'rb') as f:
            best_params = pickle.load(f)
    else:
        # Guardar best_params en un archivo
        with open(filename, 'wb') as f:
            pickle.dump(best_params, f)
    return best_params



def optimize_parameters(pick_func, nsta_values, nlta_values, thr_on_values, thr_off, stations, ventana_10s, ventana_30s, v_P, coord_list, filename):
    
    """
    Optimiza los parámetros de P_picking y guarda los mejores parámetros en un archivo.

    Entradas:
    ----------
    pick_func: str
        Función de picking a optimizar.
    nsta_values: list
        Lista con los valores de nsta.
    nlta_values: list
        Lista con los valores de nlta.
    thr_on_values: list
        Lista con los umbrales de activación a testear
    thr_off: float
        Umbral de desactivación del picking.
    stations: list
         Lista con las estaciones en formato Stream.
    ventana_10s: int
        Tiempo de desplazamiento de la ventana en segundos.
    ventana_30s: int
        Tiempo de la señal en segundos donde se aplica el STA/LTA.
    v_P: float
         Velocidad de la onda P.
    coord_list: list
        Lista con las coordenadas de las estaciones.
    filename: str
        Nombre del archivo con el cual se comparan los tiempos de detección para el cálculo de F1-score.

    Salidas:
    ----------
    best_params: dict
        Diccionario con los mejores parámetros.
    
    """

    best_params = {}

    # Número total de iteraciones
    total = len(nsta_values) * len(nlta_values) * len(thr_on_values)

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Saltar la cabecera
        tiempos_reales = [datetime.strptime(row[0], '%Y-%m-%dT%H:%M:%S') for row in reader]

    with ThreadPoolExecutor() as executor:
        if pick_func in ['p_picking', 'p_picking_val', 'p_picking_each']:
            func = globals()[pick_func]  # or locals()[pick_func] if the function is defined locally
            futures = {}
            for nsta, nlta, thr_on in product(nsta_values, nlta_values, thr_on_values):
                if pick_func == 'p_picking' or pick_func == 'p_picking_val':
                    futures[executor.submit(func, stations, ventana_10s, ventana_30s, nsta, nlta, v_P, coord_list, thr_on, thr_off)] = (nsta, nlta, thr_on)
                elif pick_func == 'p_picking_each':
                    futures[executor.submit(func, stations, ventana_10s, ventana_30s, nsta, nlta, thr_on, thr_off)] = (nsta, nlta, thr_on)
        else:
            raise ValueError('La elección debe ser "p_picking", "p_picking_val" o "p_picking_each"')



        for future in tqdm(as_completed(futures), total = total, leave = True):
            nsta, nlta, thr_on = futures[future]
            try:
                time_all = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % ((nsta, nlta, thr_on), exc))

            else:
                # Guardar los tiempos predichos
                tiempos_predichos = [datetime.strptime(str(time), '%Y-%m-%dT%H:%M:%S.%fZ') for time in time_all]
                # Calcular los conjuntos de tiempos reales y predichos
                conjunto_reales = set([t.replace(second=0) for t in tiempos_reales])
                conjunto_predichos = set([t.replace(second=0, microsecond=0) for t in tiempos_predichos]) 
                verdaderos_positivos = conjunto_reales & conjunto_predichos
                falsos_positivos = conjunto_predichos - conjunto_reales
                falsos_negativos = conjunto_reales - conjunto_predichos
                if verdaderos_positivos:
                    presicion = len(verdaderos_positivos) / (len(verdaderos_positivos) + len(falsos_positivos))
                    recall = len(verdaderos_positivos) / (len(verdaderos_positivos) + len(falsos_negativos))
                    f1_score = 2 * (presicion * recall) / (presicion + recall)
                else:
                    f1_score = 0  # Si no hay verdaderos positivos ni falsos positivos, la precisión es 0
                # Si el resultado actual es mejor que el mejor resultado hasta ahora, actualizar los mejores parámetros y el mejor resultado
                # SI estas trabajando con el catalogo de superiores a 4 grados, usamos precision como métrica 
                if filename == 'times_events_24hrs_sup40.txt':
                    if len(best_params) < 5 or presicion > min(best_params.values()):
                        if len(best_params) == 5:
                            # eliminar la combinación de parámetros con el peor ref1_score sultado
                            worst_key = min(best_params, key=best_params.get)
                            del best_params[worst_key]   
                        # agregar la nueva combinación de parámetros y su resultado al diccionario
                        best_params[(nsta, nlta, thr_on, thr_off)] = f1_score 
                        #ic(best_params)

                else:
                
                    if len(best_params) < 5 or f1_score > min(best_params.values()):
                        if len(best_params) == 5:
                            # eliminar la combinación de parámetros con el peor resultado
                            worst_key = min(best_params, key=best_params.get)
                            del best_params[worst_key]   
                        # agregar la nueva combinación de parámetros y su resultado al diccionario
                        best_params[(nsta, nlta, thr_on, thr_off)] = f1_score 
                        #ic(best_params)
    return best_params


    



# def optimize_parameters(nsta_values, nlta_values, thr_on_values, thr_off, stations, ventana_10s, ventana_30s, v_P, coord_list):
#     # Diccionario para almacenar los mejores parámetros y el resultado de f1-score
#     best_params = {}

#     # Número total de iteraciones
#     total = len(nsta_values) * len(nlta_values) * len(thr_on_values)

#     pbar = tqdm(total=total, leave=True)

#     with open('times_events_24hrs.txt', 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # Saltar la cabecera
#         tiempos_reales = [datetime.strptime(row[0], '%Y-%m-%dT%H:%M:%S') for row in reader]

#     # se itera sobre los parámetros
#     for nsta, nlta, thr_on in product(nsta_values, nlta_values, thr_on_values):
#         # Llamar a la función P_picking con los parámetros actuales
#         time_all = P_picking1(stations, ventana_10s, ventana_30s, nsta, nlta, v_P, coord_list, thr_on, thr_off)

#         # Guardar los tiempos predichos
#         tiempos_predichos = [datetime.strptime(str(time), '%Y-%m-%dT%H:%M:%S.%fZ') for time in time_all]
        
#         # Calcular los conjuntos de tiempos reales y predichos
#         conjunto_reales = set([t.replace(second=0) for t in tiempos_reales])
#         conjunto_predichos = set([t.replace(second=0, microsecond=0) for t in tiempos_predichos])
        
#         # Calcular los verdaderos positivos, falsos positivos y falsos negativos
#         verdaderos_positivos = conjunto_reales & conjunto_predichos
#         falsos_positivos = conjunto_predichos - conjunto_reales
#         falsos_negativos = conjunto_reales - conjunto_predichos
        
#         # Usar F1-score como métrica de evaluación para minimizar los falsos positivos y falsos negativos
#         if verdaderos_positivos:
#             presicion = len(verdaderos_positivos) / (len(verdaderos_positivos) + len(falsos_positivos))
#             recall = len(verdaderos_positivos) / (len(verdaderos_positivos) + len(falsos_negativos))
#             f1_score = 2 * (presicion * recall) / (presicion + recall)
#         else:
#             f1_score = 0  # Si no hay verdaderos positivos ni falsos positivos, la precisión es 0

#         # Si el resultado actual es mejor que el mejor resultado hasta ahora, actualizar los mejores parámetros y el mejor resultado
#         if len(best_params) < 5 or f1_score > min(best_params.values()):
#             if len(best_params) == 5:
#                 # eliminar la combinación de parámetros con el peor resultado
#                 worst_key = min(best_params, key=best_params.get)
#                 del best_params[worst_key]   
#             # agregar la nueva combinación de parámetros y su resultado al diccionario
#             best_params[(nsta, nlta, thr_on, thr_off)] = f1_score 
#             ic(best_params)

#         # Actualizar la barra de progreso
#         pbar.update()

#     pbar.close()

#     return best_params
