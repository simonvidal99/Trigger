import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.signal import butter, lfilter

def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def save_wav(filename, data, sampling_rate):
    wavfile.write(filename, sampling_rate, data.astype(np.int16))

def main():
    # Cargar señal de voz desde un archivo OGG
    audio_path = 'sample.ogg'
    audio = AudioSegment.from_file(audio_path, format="ogg")
    original_signal = np.array(audio.get_array_of_samples())
    sampling_rate = audio.frame_rate

    # Simular añadiendo ruido a la señal original
    noise_amplitude = 0.001 
    noise = np.random.normal(0, noise_amplitude, len(original_signal))
    noisy_signal = original_signal + noise

    # Parámetros para el filtro pasa-bajos
    cutoff_frequency = 1000  # Hz (ajustado para conservar más frecuencias)
    filter_order = 10 # Mayor orden para una mayor selectividad

    # Aplicar el filtro pasa-bajos
    filtered_signal = butter_lowpass_filter(noisy_signal, cutoff_frequency, sampling_rate, order=filter_order)

    # Guardar la señal filtrada en un nuevo archivo WAV
    save_wav('filtered_signal1.wav', filtered_signal, sampling_rate)

if __name__ == "__main__":
    main()
