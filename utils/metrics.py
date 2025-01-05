# Plotting Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def display_conf_matrix(y_true,y_pred,num_words=50):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_words))
    disp.plot(cmap="viridis")  # Customize colormap if needed
    return cm  # returning cm because there are too many classes and the Confusion Matrix gets messy.
               # cm is a numpy array and can be used for inference later

def show_performance_scores(y_true, y_pred):
    types= ['micro', 'macro', 'weighted']
    for ty in types:
        print(ty)
        precision = precision_score(y_true, y_pred, average=ty) 
        recall = recall_score(y_true, y_pred, average=ty)        
        f1 = f1_score(y_true, y_pred, average=ty) 
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("***************")
    print(f"Acc = {accuracy_score(y_true, y_pred):.4f}")
