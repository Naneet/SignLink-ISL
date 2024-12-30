# Plotting Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import confusion_matrix

def display_conf_matrix(y_true,y_pred,num_words=50):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_words))
    disp.plot(cmap="viridis")  # Customize colormap if needed
