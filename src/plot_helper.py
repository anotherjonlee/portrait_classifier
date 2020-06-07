def loss_plotter(model_history):
    """
    The function takes a trained keras model and plots a training and validating losses over epochs
    
    input:  trained keras model
    output: seaborn line plot (matplotlib object)
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    training_loss = model_history.history['loss']
    
    test_loss = model_history.history['val_loss']

    epoch_count = range(1,len(training_loss)+1)
    
    fig,ax = plt.subplots(figsize=(15,8))

    sns.set(font_scale=1.15)
    
    sns.lineplot(
        x=epoch_count,
        y=training_loss,
        ax=ax
    )
    
    sns.lineplot(
        x=epoch_count,
        y=test_loss,
        ax=ax
    )

    ax.set_title(
        'Loss Curves: Pre-Trained VGG-16 with 2 Trained Layers',
        fontsize=19
    )
    ax.set_ylabel(
        'Loss',
        fontsize=18
    )
    ax.set_xlabel(
        'Epochs',
        fontsize=18
    )

    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()
    
def acc_plotter(model_history):
    """
    The function takes a trained keras model and plots a training and validating accuracies over epochs
    
    input:  trained keras model
    output: seaborn line plot (matplotlib object)
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    acc = model_history.history['accuracy']
    
    val_acc = model_history.history['val_accuracy']

    epoch_count = range(1,len(training_loss)+1)

    fig,ax = plt.subplots(figsize=(15,8))
    
    sns.set(font_scale=1.15)
    
    sns.lineplot(
        x=epoch_count,
        y=acc,
        ax=ax
    )
    
    sns.lineplot(
        x=epoch_count,
        y=val_acc,
        ax=ax
    )

    ax.set_title('Accuracy Curves: Pre-Trained VGG-16 with 2 Trained Layers',fontsize=19)
    
    ax.set_ylabel('Accuracy',fontsize=18)
    
    ax.set_xlabel('Epochs',fontsize=18)

    plt.legend(['Training Accuracy', 'Validation Accuracy'])

    plt.show()
    
def cm_plotter(y_true, y_pred, label):
    """
    The function generates a confusion matrix from the inputed true and predicted y values, and their labels.
    
    input:  y_true and y_pred, label (arrays)
    output: confusion matrix (matplotlib object)
    """
    from sklearn.metrics import confusion_matrix   
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(
        y_true,
        y_pred, 
        normalize='true'
    )
    
    fig,ax = plt.subplots(figsize(12,12))
    
    sns.heatmap(
        cm,
        cmap='viridis',
        ax=ax,
        xticklabels=class_labels, 
        yticklabels=class_labels
    )
    
    ax.set_title(
        'Confusion Matrix of the Model', 
        fontsize = 20
    )
    
    ax.set_xlabel(
        'True Labels',
        fontsize = 18
    )
    
    ax.set_ylabel(
        'Predicted labels',
        fontsize = 18
    )
    
    plt.show()