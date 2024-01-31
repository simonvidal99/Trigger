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
    plt.tight_layout()
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


def plot_confusion_matrix(threshold, station, title, labels, log_data, classes, ax1=None, ax2=None):

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

    plt.show()

