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



def plot_roc_curve(labels, station, data, class_type, title='ROC Curve'):
    original_setting = plt.rcParams['figure.constrained_layout.use']
    plt.rcParams['figure.constrained_layout.use'] = True

    fpr, tpr, _ = roc_curve(labels, data)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(9, 5))

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)

    # linea diagonal
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{title}. Station {station}. Classes {class_type}', fontsize=14)
    
    # grid
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(loc="lower right", fontsize=10)

    # grid más detallada
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.3)

    # Show the plot
    plt.show()



def calculate_optimal_threshold(labels, data, method='youden_index'):
    fpr, tpr, thresholds = roc_curve(labels, data)

    if method == 'youden_index':
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    elif method == 'euclidena_distance':
        distances = np.sqrt((0 - fpr)**2 + (1 - tpr)**2)
        index = np.argmin(distances)
        optimal_threshold = thresholds[index]
    elif method == 'concordance_probability':
        cz = tpr * (1 - fpr)
        optimal_threshold = thresholds[np.argmax(cz)]
    else:
        raise ValueError("Invalid method")

    return optimal_threshold

