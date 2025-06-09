import matplotlib.pyplot as plt
from datetime import datetime

from types import SimpleNamespace
import json

import os

##########################################
##                 json                 ##
##########################################
def load_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
        data = SimpleNamespace(**data)

    return data

##########################################
##                 PLOT                 ##
##########################################

def save_loss_plot(history, loss_filepath='loss.png', time_flag=False):
    """
    Given a Keras History object, plot & save loss curves.

    Args:
        history: keras.callbacks.History returned by model.fit()
        loss_filepath (str): where to save the loss plot (PNG).
    """
    if not time_flag:
        # Plot training & validation loss
        dirname = './graph/'
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        loss_filepath = dirname+loss_filepath

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

    else:
        # Plot training & validation loss
        dirname = './graph/'+name_date('./graph')
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        name, ext = loss_filepath.split('.')
        loss_filepath = dirname+'/'+name_time(name)+"."+ext

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

def save_acc_plot(history, acc_filepath='accuracy.png', time_flag=False):
    '''
    Given a Keras History object, plot & save accuracy curves.

    Args:
        history: keras.callbacks.History returned by model.fit()
        acc_filepath (str): where to save the accuracy plot (PNG).
    '''
    if not time_flag:
        # Plot training & validation accuracy
        # try both 'accuracy' and 'acc' keys for compatibility
        dirname = './graph/'
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        acc_filepath = dirname+acc_filepath

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
    
    else:
        # Plot training & validation loss
        dirname = './graph/'+name_date('./graph')
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        name, ext = acc_filepath.split('.')
        acc_filepath = dirname+'/'+name_time(name)+"."+ext

        # Plot training & validation accuracy
        # try both 'accuracy' and 'acc' keys for compatibility
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

def name_date(default_name,ext=None):
    if ext == None:
        now = datetime.now()
        time_str = now.strftime('%y%m%d')
        name = default_name+'_'+time_str
    else:
        now = datetime.now()
        time_str = now.strftime('%y%m%d')
        name = default_name+'_'+time_str+ext

    return name

def name_time(default_name,ext=None):
    if ext == None:
        now = datetime.now()
        time_str = now.strftime('%H%M%S')
        name = default_name+'_'+time_str
    else:
        now = datetime.now()
        time_str = now.strftime('%y%m%d')
        name = default_name+'_'+time_str+ext
    
    return name