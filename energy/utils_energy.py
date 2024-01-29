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
from astropy.visualization import hist



def nearest_station(file_path: str, stations_names:list):
    '''
     Creamos una lista con el tiempo de partida de cada evento para la estación más cercana y una lista con el nombre de la estación más cercana para cada evento
    '''

    # Lee los datos
    df = pd.read_csv(file_path)

    results = {}

    # Itera sobre cada fila del DataFrame
    for i, fila in df.iterrows():
        
        # Obtiene las horas de detección en las estaciones. Esto se debe cambiar según las estaciones que tenga para analizar
        horas_deteccion = [fila['Inicio_CO10'], fila['Inicio_AC04'], fila['Inicio_AC05'], fila['Inicio_CO05']]
        
        # Encuentra el índice de la estación más cercana
        indice_estacion_cercana = horas_deteccion.index(min(horas_deteccion))
        
        # Guarda la estación más cercana y la hora de detección correspondiente en el diccionario
        results[i+1] = [stations_names[indice_estacion_cercana], horas_deteccion[indice_estacion_cercana]]

    
    start_time = [UTCDateTime(results[clave][1]) for clave in sorted(results)]
    closest_st_names = [results[clave][0] for clave in sorted(results)]

    return start_time, closest_st_names


def nearest_two_stations(df: pd.DataFrame, stations_names:list):
    '''
     Creamos una lista con el tiempo de partida de cada evento para las dos estaciones más cercanas y una lista con los nombres de las dos estaciones más cercanas para cada evento
    '''

    # Lee los datos
    #df = pd.read_csv(df)

    results = {}

    # Itera sobre cada fila del DataFrame
    for i, fila in df.iterrows():
        
        # Obtiene las horas de detección en las estaciones. Esto se debe cambiar según las estaciones que tenga para analizar
        horas_deteccion = [fila['Inicio_CO10'], fila['Inicio_AC04'], fila['Inicio_AC05'], fila['Inicio_CO05']]
        
        # Encuentra los índices de las dos estaciones más cercanas
        indices_estaciones_cercanas = sorted(range(len(horas_deteccion)), key=lambda k: horas_deteccion[k])[:2]
        
        # Guarda las dos estaciones más cercanas y las horas de detección correspondientes en el diccionario
        results[i+1] = [[stations_names[indice] for indice in indices_estaciones_cercanas], [horas_deteccion[indice] for indice in indices_estaciones_cercanas]]

    start_times = [[UTCDateTime(results[clave][1][j]) for clave in sorted(results)] for j in range(2)]
    closest_st_names = [[results[clave][0][j] for clave in sorted(results)] for j in range(2)]

    return start_times, closest_st_names




def signal_energy(signal, frame_size = 160, sample_rate = 40, hop_lenght = 160):

    '''

    Calculates the energy of a signal in different frames
    ----------------------------------------------

    Inputs:
    - signal: np.array()
        Signal where the energy calculation is being performed
    - frame_size: int
        Lenght of the desired frame in samples 
    - sample_rate: int
        Sample rate of the signal 
    - hop_lenght: int
        In case we wanted to have overlap, this value represents the "jump" of the frame.
        Ex: If hop_lenght = 80, after calculating the energy in the first frame (first 4 seconds), the next energy calculation would not
        be in the next 4 seconds, but it would start in the middle of the first frame instead. 

    Outputs:
    - energy: np.array()
        Contains the value of the energy for each frame

    
    '''

    # Acá tomo los primeros 60 segundos porque no puedo tomar tomar todo el resto de la traza (serían horas y chocaria con otros eventos).
    # En el caso de tiempo real esto no sería necesario ya que ya habría solamente un tiempo determinado de la traza siendo examinado.
    signal = signal[:60*sample_rate]
    
    # Multiplicamos por el factor de escala dado en el paper de estimación de magnitud  
    data = signal*1e10

    # Calcular la energía en frames de 4 segundos 
    energy = np.convolve(data**2, np.ones(frame_size), mode='valid')[::hop_lenght]

    # Lo siguiente hace lo mismo, pero es MUCHO más lento
    #energy = np.array([
    #    np.sum(np.abs(data[i:i+window_size])**2)
    #    for i in range(0, len(data), hop_length)
    #])

    return energy


def endpoint_event(signal, thr_energy = 0.03, frame_size = 160):

    '''
    Calculates where the event is over by identifying in which frame the energy is lower than 3% of the peak energy
    ----------------------------------------------

    Inputs:
    - signal: np.array()
        Signal
    - thr_energy: float
        Percentage to be used as threshold
    - frame_size: int
        Lenght of the desired frame in samples 

    Outputs:
    - peak_index_energy: 
        Sample where the max energy is achieved
    - endpoint_energy:
        Sample where the energy goes bellow 3%
    '''

    energy = signal_energy(signal)

    # Finds the framw with the maximun energy
    max_energy = np.max(energy)
    peak_index_energy = np.argmax(energy)
    #ic(peak_index_energy)

    # Calculates the 3% of the peak energy 
    threshold_energy = thr_energy * max_energy

    # Find the point where the signal drops below 3% after the peak.
    # The following condition is in case it does not find within the minute (or the time defined in the function signal_energy) 
    # a frame with energy less than 3%, in this case it simply returns the last frame
    index = np.where(energy[peak_index_energy:] < threshold_energy)[0]
    if index.size > 0:
        endpoint_energy = index[0] + peak_index_energy
    else:
        endpoint_energy = len(energy) - 1  # or any other value that makes sense in your context

    #endpoint_energy = np.where(energy[peak_index_energy:] < threshold_energy)[0][0] + peak_index_energy
    #ic(endpoint_energy)

    endpoint_energy = endpoint_energy*frame_size  #porel largo del frame ya que endpoint energy entrega el frame donde se bajo del 3%, entonces necesitamos esto para tenerlo en muestras
    return peak_index_energy, endpoint_energy
    



def energy_power(signal, window_size = 160, sample_rate = 40, hop_lenght = 160):

    # Multiplicamos por el factor multiplicado en el paper de estimación de magnitud
    data = signal*1e10

    # Calcular la energía en frames de 4 segundos
    energy = np.convolve(data**2, np.ones(window_size), mode='valid')[::hop_lenght]
    #energy = np.array([
    #    sum(abs(data[i:i+window_size]**2))
    #    for i in range(0, len(data), window_size)
    #])

    # Calcular la potencia como la tasa de cambio de energía con respecto al tiempo
    #power = energy / (window_size / sample_rate) #basicamente energía dividio en el tiempo de cada frame. En este caso el frame sería el window_size
    power = np.cumsum(energy) / ((np.arange(len(energy)) + 1) * window_size / sample_rate) # se calcula la potencia como la suma de la energía dividida en el tiempo

    return energy, power



def plot_power(power_events, station, n_frames=1, use_log=False, height=6, width=4, event_type=None, use_mean=False,ax=None):

    original_setting = plt.rcParams['figure.constrained_layout.use']
    plt.rcParams['figure.constrained_layout.use'] = True


    #plt.figure(figsize=(height, width))

    if ax is None:
        fig, ax = plt.subplots(figsize=(height, width)
                               )
    else:
        fig = ax.figure
        fig.set_constrained_layout(True)

    fig.set_constrained_layout(True)

    # Definimos una lista de colores
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Encontrar la longitud máxima del array más largo en power_events
    max_len = max(max(len(eventos) for eventos in events) for events in power_events)

    for i, events in enumerate(power_events):

        if use_mean:
            # En caso de querer considerar el hecho de tomar más frames luego de que el evento haya finalizado como parte del cálculo de la potencia
            # (dudo, ya que tenemos el criterio del 3% de la energía para saber si un evento terminó), paddeamos con zeros los arrays más pequeños para que 
            # tengan suficientes frames para "cubrir" n_frames y para el cálculo de la potencia ahora estos 0's se consideran. Es decir, la potencia pasa a ser ahora
            # el último valor del array (potencia del último frame que sería la potencia de la señal, energía total sobre el tiempo del evento) promediado con la cantidad
            # de ceros que hayan de padding.
            first_n_frames = [np.mean(np.pad(eventos, (0, max_len - len(eventos)), 'constant')[:n_frames]) if len(eventos) < max_len else eventos[n_frames-1] for eventos in events]
        else:
            # se toma el frame pedido o bien el último frame si es que el frame que se ingresó como parámetro supera la cantidad existente
            first_n_frames = [eventos[min(n_frames-1, len(eventos)-1)] for eventos in events]

        if use_log:
            first_n_frames = np.log10(first_n_frames)

        # plt.hist(
        #     first_n_frames,
        #     bins=40,  # Aumenta la cantidad de bins
        #     edgecolor='black',
        #     color=colors[i % len(colors)],
        #     alpha=0.6,  # Ajusta la transparencia
        #     label=event_type[i] if event_type else None
        # )
            bins=['knuth']
            ax.hist(first_n_frames, 
                 bins = 10,
                 edgecolor='black', 
                 color=colors[i % len(colors)], 
                 #histtype='stepfilled',
                 alpha=0.6, 
                 label=event_type[i] if event_type else None,
                 density=False)

    title = 'Potencia de los primeros {} frames. Station {}'.format(n_frames, station)
    xlabel = 'Potencia'
    if use_log:
        title = 'Log de la ' + title
        xlabel = 'Log de la ' + xlabel

    ax.set_xlim([4, 16])
    ax.set_ylim([0, 50])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cantidad')

    # Añadimos la leyenda si event_type no es None
    if event_type:
        ax.legend()
    plt.rcParams['figure.constrained_layout.use'] = original_setting

    #plt.show()



def plot_power_each(power_events, station, n_frames=1, use_log=False, height = 6, width = 4, event_type=''):


    original_setting = plt.rcParams['figure.constrained_layout.use']
    plt.rcParams['figure.constrained_layout.use'] = True
    # Extraemos los primeros n elementos de cada array y calculamos su promedio
    first_n_frames = [np.mean(evento[:n_frames]) for evento in power_events]

    # Aplicamos la transformación logarítmica a los datos si use_log es True
    if use_log:
        first_n_frames = np.log10(first_n_frames)

    # Creamos una figura con un tamaño específico
        
    plt.figure(figsize=(height, width))

    # Creamos el histograma
    #plt.hist(first_n_frames, bins=10, edgecolor='black')

    hist(first_n_frames, 
        bins='knuth',
        edgecolor='black', 
        #color=colors[i % len(colors)], 
        #histtype='stepfilled',
        alpha=0.6, 
        #label=event_type[i] if event_type else None,
        density=False)

    # Agregamos títulos y etiquetas
    title = 'Potencia de los primeros {} frames {}. Estación {}'.format(n_frames, event_type, station)
    xlabel = 'Potencia'
    if use_log:
        title = 'Log de la ' + title
        xlabel = 'Log de la ' + xlabel

    #plt.xlim([4,15])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frecuencia')

    # Mostramos el gráfico
    #plt.show()


# def plot_energy_hist(energy_events, station, frame=1, use_log=False, height = 6, width = 4, event_type=None):


#     plt.figure(figsize=(height, width))

#     # Definimos una lista de colores
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

#     # Encontrar la longitud máxima del array más largo en power_events
#     max_len = max(max(len(eventos) for eventos in events) for events in energy_events)

#     for i, events in enumerate(energy_events):

#         energy_frame = [eventos[frame-1] if len(eventos) > frame-1 else 1 for eventos in events]

#         if use_log:
#             energy_frame = np.log10(energy_frame)

#         #plt.hist(energy_frame, bins=10, edgecolor='black', color=colors[i % len(colors)], alpha=0.5, label=event_type[i] if event_type else None)
#         bins = ['knuth']
#         hist(energy_frame, 
#             bins= 10, 
#             edgecolor='black', 
#             color=colors[i % len(colors)], 
#             #histtype='stepfilled',
#             alpha=0.6, 
#             label=event_type[i] if event_type else None,
#             density=False)
               

#     title = 'Energía en el frame {}. Estación {}'.format(frame, station)
#     xlabel = 'Energía'
#     if use_log:
#         title = 'Log de la ' + title
#         xlabel = 'Log de la ' + xlabel

#     plt.xlim([4,15])
#     plt.ylim([0,50])
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel('Frecuencia')

#     # Añadimos la leyenda si event_type no es None
#     if event_type:
#         plt.legend()


#     plt.show()

def plot_energy_hist(energy_events, station, frame=1, use_log=False, event_type=None, ax=None):

    original_setting = plt.rcParams['figure.constrained_layout.use']
    plt.rcParams['figure.constrained_layout.use'] = True

    # If no ax was provided, create a new figure and ax
    if ax is None:
        fig, ax = plt.subplots()

    # Definimos una lista de colores
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Encontrar la longitud máxima del array más largo en power_events
    max_len = max(max(len(eventos) for eventos in events) for events in energy_events)

    for i, events in enumerate(energy_events):

        energy_frame = [eventos[frame-1] if len(eventos) > frame-1 else 1 for eventos in events]

        if use_log:
            energy_frame = np.log10(energy_frame)

        # Use ax.hist instead of plt.hist
        bins = ['knuth']
        ax.hist(energy_frame, 
            bins= 10, 
            edgecolor='black', 
            color=colors[i % len(colors)], 
            alpha=0.6, 
            label=event_type[i] if event_type else None,
            density=False)

    title = 'Energía en el frame {}. Estación {}'.format(frame, station)
    xlabel = 'Energía'
    if use_log:
        title = 'Log de la ' + title
        xlabel = 'Log de la ' + xlabel

    ax.set_xlim([4,15])
    ax.set_ylim([0,50])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frecuencia')

    # Añadimos la leyenda si event_type no es None
    if event_type:
        ax.legend()

    # No need to call plt.show() here, do it after you've created all your subplots


def plot_confusion_matrix(threshold, station, title, labels, log_data, classes, ax1=None, ax2=None):
    # Disable constrained layout for the confusion matrix
    original_setting = plt.rcParams['figure.constrained_layout.use']
    plt.rcParams['figure.constrained_layout.use'] = False
    predicted_labels = np.array([1 if x >= threshold else 0 for x in log_data])

    cm = confusion_matrix(labels, predicted_labels)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the original confusion matrix
    im1 = ax1.imshow(cm, cmap='coolwarm')
    ax1.set_title(f'CM {title}. Station {station}', fontsize=10)
    ax1.set_xlabel('Predicted', fontsize=8)
    ax1.set_ylabel('True', fontsize=8)

    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax1.text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=6, fontweight='bold')

    cbar1 = plt.colorbar(im1, ax=ax1)

    ax1.set_xticks(np.arange(len(classes)))
    ax1.set_yticks(np.arange(len(classes)))
    ax1.set_xticklabels(classes, fontsize=8)
    ax1.set_yticklabels(classes, fontsize=8)

    # Plot the normalized confusion matrix
    im2 = ax2.imshow(cm_normalized, cmap='coolwarm', vmin=0, vmax=1)
    ax2.set_title(f'Normalized CM {title}', fontsize=10)
    ax2.set_xlabel('Predicted', fontsize=8)
    ax2.set_ylabel('True', fontsize=8)

    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax2.text(j, i, f'{cm_normalized[i, j]*100:.2f}%', ha='center', va='center', color='white', fontsize=6, fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=ax2)

    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes, fontsize=8)
    ax2.set_yticklabels(classes, fontsize=8)

    return ax1, ax2  # Return the original AxesSubplot objects


def plot_confusion_matrix2(threshold, station, title, labels, log_data, classes):
    # Disable constrained layout for the confusion matrix
    original_setting = plt.rcParams['figure.constrained_layout.use']
    plt.rcParams['figure.constrained_layout.use'] = False
    predicted_labels = np.array([1 if x >= threshold else 0 for x in log_data])

    cm = confusion_matrix(labels, predicted_labels)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axs = plt.subplots(1,2, figsize = (10,4))

    # Plot the original confusion matrix
    im1 = axs[0].imshow(cm, cmap='coolwarm')
    axs[0].set_title(f'CM {title}. Station {station}')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('True')

    for i in range(len(classes)):
        for j in range(len(classes)):
            text = axs[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontsize=14, fontweight='bold')

    cbar1 = plt.colorbar(im1, ax=axs[0])

    axs[0].set_xticks(np.arange(len(classes)))
    axs[0].set_yticks(np.arange(len(classes)))
    axs[0].set_xticklabels(classes)
    axs[0].set_yticklabels(classes)

    # Plot the normalized confusion matrix
    im2 = axs[1].imshow(cm_normalized, cmap='coolwarm', vmin=0, vmax=1)
    axs[1].set_title(f'Normalized CM {title}')
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('True')

    for i in range(len(classes)):
        for j in range(len(classes)):
            text = axs[1].text(j, i, f'{cm_normalized[i, j]*100:.2f}%', ha='center', va='center', color='white', fontsize=14, fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=axs[1])

    axs[1].set_xticks(np.arange(len(classes)))
    axs[1].set_yticks(np.arange(len(classes)))
    axs[1].set_xticklabels(classes)
    axs[1].set_yticklabels(classes)

    plt.tight_layout()
    plt.show()
    plt.rcParams['figure.constrained_layout.use'] = True

if __name__ == '__main__':
    
    print('que pasa jiles culiaooos')