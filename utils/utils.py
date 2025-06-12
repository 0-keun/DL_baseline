import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from types import SimpleNamespace
import json

import os

##########################################
##                 json                 ##
##########################################
def load_json(file_name='./params.json'):
    with open(file_name, 'r') as f:
        data = json.load(f)
        data = SimpleNamespace(**data)

    return data

##########################################
##                 PLOT                 ##
##########################################

def save_loss_plot(history, loss_filename='loss.png', time_flag=False):
    """
    Given a Keras History object, plot & save loss curves.

    Args:
        history: keras.callbacks.History returned by model.fit()
        loss_filepath (str): where to save the loss plot (PNG).
    """
    loss_filepath = name_to_dir(name='graph',time_flag=time_flag)+name_time(default_name=loss_filename)
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_filepath, dpi=150, bbox_inches='tight')
    plt.close()

def save_acc_plot(history, acc_filename='accuracy.png', time_flag=False):
    '''
    Given a Keras History object, plot & save accuracy curves.

    Args:
        history: keras.callbacks.History returned by model.fit()
        acc_filepath (str): where to save the accuracy plot (PNG).
    '''
    acc_filepath = name_to_dir(name='graph',time_flag=time_flag)+name_time(default_name=acc_filename)
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_' + acc_key
    plt.figure()
    plt.plot(history.history[acc_key], label='train acc')
    if val_acc_key in history.history:
        plt.plot(history.history[val_acc_key], label='val acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_filepath, dpi=150, bbox_inches='tight')
    plt.close()

def get_confusion_mat(y_true, y_pred, name='confusion_matrix', time_flag=False , save_csv=True, save_png=True):
    csv_filepath = name_to_filepath(name_ext=name+'.csv',time_flag=time_flag)
    png_filepath = name_to_filepath(name_ext=name+'.png',time_flag=time_flag)

    cm=confusion_matrix(y_true, y_pred)
    if save_png:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(png_filepath)
    if save_csv:
        np.savetxt(csv_filepath, cm, fmt='%d', delimiter=',')


##############################
##           NAME           ##
##############################

def name_date(default_name,ext=None):
    now = datetime.now()
    time_str = now.strftime('%y%m%d')

    if ext == None:
        if '.' in default_name:
            name, ext = default_name.split('.')
            ext = '.'+ext
            name = name+'_'+time_str+ext
        else:
            name = default_name+'_'+time_str
    else:
        name = default_name+'_'+time_str+ext

    return name

def name_time(default_name,ext=None):
    now = datetime.now()
    time_str = now.strftime('%H%M%S')
    if ext == None:
        if '.' in default_name:
            name, ext = default_name.split('.')
            ext = '.'+ext
            name = name+'_'+time_str+ext
        else:
            name = default_name+'_'+time_str
    else:
        name = default_name+'_'+time_str+ext
    
    return name

def name_to_filepath(name_ext, time_flag=False):
    '''
    return dirname+filename using name_ext
    '''
    if '.' in name_ext:
        name, ext = name_ext.split('.')
        ext = '.'+ext

    if not time_flag:
        dirname = './'+name+'/'
        filename = name+'.'+ext
    else:
        dirname = './'+name+'/'+name_date(name)+'/'
        filename = name_time(name)+'.'+ext
    
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    
    return dirname+filename

def name_to_dir(name, time_flag=False):
    if not time_flag:
        dirname = './'+name+'/'
    else:
        dirname = './'+name+'/'+name_date(name)+'/'
    
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    
    return dirname
