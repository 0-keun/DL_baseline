import matplotlib.pyplot as plt


##########################################
##                 PLOT                 ##
##########################################

def save_loss_plot(history, loss_filepath='loss.png'):
    """
    Given a Keras History object, plot & save loss curves.

    Args:
        history: keras.callbacks.History returned by model.fit()
        loss_filepath (str): where to save the loss plot (PNG).
    """
    # Plot training & validation loss
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

def save_acc_plot(history, acc_filepath='accuracy.png'):
    '''
    Given a Keras History object, plot & save accuracy curves.

    Args:
        history: keras.callbacks.History returned by model.fit()
        acc_filepath (str): where to save the accuracy plot (PNG).
    '''
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