# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from obspy.signal.trigger import classic_sta_lta

# Generamos una señal de prueba más larga con algunos picos
np.random.seed(1)
a = np.random.rand(2500)

# Simulamos una señal sísmica con varios picos
for _ in range(10):
    start = np.random.randint(0, len(a) - 500)
    a[start:start+100] += np.sin(np.linspace(0, 1, 100)) * np.random.uniform(0.5, 1.5)

# Parámetros STA/LTA en segundos
nsta = 2
nlta = 10
fs = 40  # Frecuencia de muestreo en Hz

# Convertir nsta y nlta de segundos a muestras
nsta_samples = int(nsta * fs)
nlta_samples = int(nlta * fs)

# Luego pasas estos valores a la función classic_sta_lta
cf = classic_sta_lta(a, nsta_samples, nlta_samples)

# Calculamos el tiempo correspondiente en segundos
time_seconds = np.arange(len(a)) / fs

# Creamos la figura y los ejes para la animación
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Líneas para almacenar los objetos de línea para actualizar
line1, = ax1.plot([], [], color='b', label='Seismic Trace', linewidth=0.3)  # Hacemos la línea más fina
line2, = ax2.plot([], [], color='r', label='STA/LTA', linewidth=2)

# Ventanas STA y LTA
span1 = ax1.axvspan(0, 0, alpha=0.3, color='red', label='LTA window')
span2 = ax1.axvspan(0, 0, alpha=0.3, color='green', label='STA window')

# Ventana para marcar la señal cuando el ratio STA/LTA supera el umbral
span3 = ax1.axvspan(0, 0, alpha=0.3, color='yellow', label='Seismic event')

# Texto para el valor del ratio STA/LTA
text_ratio = ax2.text(0, 0, '', fontsize=8)

# Definimos un umbral para el ratio STA/LTA
threshold = 1.5
line3, = ax2.plot([], [], color='g', label='Threshold', linewidth=2)


def init():
    ax1.set_xlim(0, np.max(time_seconds))
    ax1.set_ylim(0, 3)
    ax1.set_title('Seismic Trace')
    ax1.set_ylabel('Amplitude')

    ax2.set_xlim(0, np.max(time_seconds))
    ax2.set_ylim(0, np.max(cf))
    ax2.set_title('STA/LTA')
    ax2.set_ylabel('Ratio')
    ax2.set_xlabel('Time (s)')

    ax1.legend()
    ax2.legend()

    return line1, line2, span1, span2, span3, text_ratio, line3

event_bands= []

def update(i):
    # Actualizar datos
    line1.set_data(time_seconds[:i+1], a[:i+1])
    line2.set_data(time_seconds[:i+1], cf[:i+1])
    line3.set_data(time_seconds, [threshold]*len(time_seconds))

    # Mostrar las ventanas STA y LTA
    if i > nlta_samples:
        span1.set_xy([[time_seconds[i-nlta_samples], 0], [time_seconds[i-nlta_samples], 1], [time_seconds[i-nsta_samples], 1], [time_seconds[i-nsta_samples], 0]] )
        span2.set_xy([[time_seconds[i-nsta_samples], 0], [time_seconds[i-nsta_samples], 1], [time_seconds[i], 1], [time_seconds[i], 0]] )

        # Mostrar el valor del ratio STA/LTA
        text_ratio.set_text(f'Ratio: {cf[i]:.2f}')
        text_ratio.set_position((time_seconds[i], cf[i]))

        # Marcar el punto en la traza sísmica si el ratio STA/LTA supera el umbral
        #if cf[i] > threshold:
            #ax1.plot(time_seconds[i], a[i], 'ro')
            #span3.set_xy([[time_seconds[i-nsta_samples], 0], [time_seconds[i-nsta_samples], 1], [time_seconds[i], 1], [time_seconds[i], 0]] )

        # Marcar el punto en la traza sísmica si el ratio STA/LTA supera el umbral
        if cf[i] > threshold:
            event_bands.append([time_seconds[i-nsta_samples], time_seconds[i]])
            for band in event_bands:
                ax1.axvspan(*band, alpha=0.3, color='yellow')

    return [line1, line2, span1, span2, text_ratio, line3] + [ax1.axvspan(*band, alpha=0.3, color='yellow') for band in event_bands]

# Creamos la animación con intervalo de 200ms para ralentizarla
ani = animation.FuncAnimation(fig, update, frames=range(len(a)), init_func=init, blit=True, interval=2)

#guardar la animación
#ani.save('animacion3.gif', writer='imagemagick', fps=30)


# Mostramos la animación
plt.show()
