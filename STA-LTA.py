# -*- coding: utf-8 -*-
"""Created on Mon Nov 20 22:31:46 2023 @author: aaron"""

import numpy as np
import numpy.matlib
from obspy import read
import matplotlib.pyplot as plt
from obspy.signal.trigger import classic_sta_lta,trigger_onset, plot_trigger
import os



def get_frames(signal, frame_length, frame_shift, window=None):
    if window is None:
        window=np.hamming(frame_length) 

    L = len(signal)
    N = int(np.fix((L-frame_length)/frame_shift + 1)) #number of frames

    Index = (np.matlib.repmat(np.arange(frame_length),N,1)+np.matlib.repmat(np.expand_dims(np.arange(N)*frame_shift,1),1,frame_length)).T
    hw = np.matlib.repmat(np.expand_dims(window,1),1,N)
    Seg = signal[Index] * hw
    return Seg.T

def E2(signal, frame_length,frame_shift,scale='logaritmica'):
    y = get_frames(signal,frame_length,frame_shift)        

    if scale =='logaritmica':
        Y = np.log10(np.sum(y**2,1))
    elif scale == 'lineal':
        Y = np.sum(y**2,1)
    return Y

def E3(signal, frame_length,frame_shift,scale='logaritmica'):
    Edos = E2(signal, frame_length,frame_shift,scale)
    if scale =='logaritmica':
        salida = Edos/np.max(Edos) #Edos-np.max(Edos)
    elif scale == 'lineal':
        salida = Edos-np.max(Edos)    
    return salida

def corte_coda(energy,umbral,fs,frame_len,frame_shi):
    arg_amp_maxima = np.argmax(energy) #Assumption: The maximun energy is in S-wave
    arg_amp_minima = np.argmin(energy[:arg_amp_maxima]) # take the minimum energy between the start of the signal and the S-wave
    delta_energia = energy[arg_amp_maxima]-energy[arg_amp_minima] 
    energia_umbral_corte = delta_energia*umbral+energy[arg_amp_minima] #energy threshold

    arg_fin_nueva_coda = arg_amp_maxima + np.argmin(np.abs(energy[arg_amp_maxima:]-energia_umbral_corte))                      
    muestra_corte_coda = int(fs*frame_len*arg_fin_nueva_coda/frame_shi)
    return muestra_corte_coda      


path_carpeta_sac = 'eventos_6-9/' #PC
lista_eventos = os.listdir(path_carpeta_sac)
print(len(lista_eventos))
# i = 3
# i = 23
i = 3
lista_estacion = os.listdir(path_carpeta_sac+'/'+lista_eventos[i])
j = 3

st = read(path_carpeta_sac+lista_eventos[i]+'/'+lista_estacion[j]+'/'+lista_estacion[j]+'_*HZ.sac')
st += read(path_carpeta_sac+lista_eventos[i]+'/'+lista_estacion[j]+'/'+lista_estacion[j]+'_*HE.sac')
st += read(path_carpeta_sac+lista_eventos[i]+'/'+lista_estacion[j]+'/'+lista_estacion[j]+'_*HN.sac')

# st.filter('highpass', freq=1.0, corners=1, zerophase=True) 
st.filter('bandpass', freqmin=4.0, freqmax=10.0, corners=1, zerophase=True) 
# print(stz[0].stats)
print(st[0].stats.channel)
# t = st[0] / 35 # division de trazas en 35 segmentos
# st[0].plot()    
# st[1].plot()   
# st[2].plot()   

# STA/LTA
fs = int(st[0].stats.sampling_rate) # Tasa de muestreo
# st[0].data = st[0].data[00:400]
cftz = classic_sta_lta(st[0].data, int(0.04 * fs), int(1 * fs))
cfte = classic_sta_lta(st[1].data, int(1 * fs), int(10 * fs))
cftn = classic_sta_lta(st[2].data, int(1 * fs), int(10 * fs))

plot_trigger(st[0],cftz, 18, 18.1)
# plot_trigger(st[1],cfte, 8, 7)
# plot_trigger(st[2],cftn, 8, 7)
# PICK ONDA P Y ONDA S

frame_p =trigger_onset(cftz, 11, 11.1)[0][0]
# frame_s =trigger_onset(cfte, 3.2, 2)[0][0]
frame_s = np.argmax(st[1].data)
time_sp = (frame_s-frame_p)/fs       

# CORTE DEL EVENTO
frame_length = 4
frame_shift = 2
frame_len = frame_length*fs
frame_shi = frame_shift*fs  
umbral_corte = 0.03

stz = st[0].data
ste = st[1].data
stn = st[2].data

E_Z_ref = E3(stz, frame_len,frame_shi,scale = 'lineal')
E_E_ref = E3(ste, frame_len,frame_shi,scale = 'lineal')
E_N_ref = E3(stn, frame_len,frame_shi,scale = 'lineal')

coda_corte = corte_coda(E_Z_ref,umbral_corte,fs,frame_len,frame_shi)      

#
plt.plot(stz, color='black')
plt.axvline(x=frame_p,color='red',dashes=[6, 2])
plt.axvline(x=frame_s,color='blue',dashes=[6, 2])
plt.axvline(x=coda_corte, color='green',dashes=[6, 2])
plt.show()