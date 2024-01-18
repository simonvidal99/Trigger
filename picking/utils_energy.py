# Standard Library Imports
import datetime

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm





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
    power = energy / (window_size / sample_rate) #basicamente energía dividio en el tiempo de cada frame. En este caso el frame sería el window_size
    #power = np.mean(energy)

    return energy, power


def plot_power(power_events, n_frames=1, use_log=False, height = 6, width = 4, event_type=None):
    # Creamos una figura con un tamaño específico
    plt.figure(figsize=(height, width))

    # Definimos una lista de colores
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, events in enumerate(power_events):
        # Extraemos los primeros n elementos de cada array y calculamos su promedio
        first_n_frames = [np.mean(evento[:n_frames]) for evento in events]

        # Aplicamos la transformación logarítmica a los datos si use_log es True
        if use_log:
            first_n_frames = np.log10(first_n_frames)

        # Creamos el histograma
        plt.hist(first_n_frames, bins=10, edgecolor='black', color=colors[i % len(colors)], alpha=0.5, label=event_type[i] if event_type else None)

    # Agregamos títulos y etiquetas
    title = 'Potencia de los primeros {} frames'.format(n_frames)
    xlabel = 'Potencia'
    if use_log:
        title = 'Log de la ' + title
        xlabel = 'Log de la ' + xlabel

    plt.xlim([4,15])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frecuencia')

    # Añadimos la leyenda si event_type no es None
    if event_type:
        plt.legend()

    # Mostramos el gráfico
    plt.show()



def plot_power_each(power_events, n_frames=1, use_log=False, height = 6, width = 4, event_type=''):
    # Extraemos los primeros n elementos de cada array y calculamos su promedio
    first_n_frames = [np.mean(evento[:n_frames]) for evento in power_events]

    # Aplicamos la transformación logarítmica a los datos si use_log es True
    if use_log:
        first_n_frames = np.log10(first_n_frames)

    # Creamos una figura con un tamaño específico
        
    plt.figure(figsize=(height, width))

    # Creamos el histograma
    plt.hist(first_n_frames, bins=10, edgecolor='black')

    # Agregamos títulos y etiquetas
    title = 'Potencia de los primeros {} frames {}'.format(n_frames, event_type)
    xlabel = 'Potencia'
    if use_log:
        title = 'Log de la ' + title
        xlabel = 'Log de la ' + xlabel

    plt.xlim([4,15])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frecuencia')

    # Mostramos el gráfico
    plt.show()