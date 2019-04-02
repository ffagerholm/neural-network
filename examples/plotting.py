import itertools
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import confusion_matrix


def plot_training_history(train_loss, test_loss, train_acc, test_acc, 
                          figsize=(12, 5), cmap=plt.cm.Dark2):
    """Plots the loss and accuracy per training epoch.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colormap = plt.get_cmap(cmap)
    ax.set_prop_cycle(cycler('color', [colormap(k) for k in np.linspace(0, 0.5, 2)]))

    ax.plot(train_loss, label='Training loss')
    ax.plot(test_loss, label='Testing loss')

    axt = ax.twinx()
    axt.set_prop_cycle(cycler('color', [colormap(k) for k in np.linspace(0.75, 1, 2)]))
    
    axt.plot(train_acc, label='Training accuracy')
    axt.plot(test_acc, label='Testing accuracy')
    axt.set_ylabel('Accuracy')

    ax.legend(loc='upper left')
    axt.legend(loc='upper right')

    ax.set_title('Loss and accuracy per iteration')
    ax.set_xlabel('Iteratio')
    ax.set_ylabel('Loss')
    
    return fig, ax


def plot_confusion_matrix(y_true, y_pred, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.RdGy, 
                          figsize=(8, 8)):
    """Plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()