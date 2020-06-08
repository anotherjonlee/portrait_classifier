
def personal_model(dim = 150, lr = 0.005):
    """
    The function takes in the dimension of the images(dim) and the learning rate(lr) for the CNN model.
    It will proceed to compile a CNN model based on the aforementioned information.
    
    Input:  dim (integer), lr (float)
    Output: compiled keras model
    """
    
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
    import data_generator

    img_width, img_height = dim, dim
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    
    else:
        input_shape = (img_width, img_height, 3)

    # Model architecture
    model = Sequential()
    # First layer
    model.add(Conv2D(32, (3, 3), input_shape=(input_shape)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), input_shape=(input_shape)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # Second layer
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # Third layer
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # Last dense layer
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12))
    model.add(Activation('softmax'))

    optimizer = SGD(lr=lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

def my_gen(gen, epoch = 50, steps_per_epoch = 100):
    
    """
    The function allows the image generator object to ignore errors when it attempts to open 
    a corrupted image.
    
    Input:  keras training ImageDataGenerator object
    """
    i=0
    
    while i < steps_per_epoch * epoch:
        try:
            data, labels = next(gen)
            i+=1
            yield data, labels
        except:
            pass
        
def model_fit(train, val, model, epoch=50, steps_per_epoch = 100):
    """
    The function will train the compiled model with the provided parameters and save the trained model.
    It will also print the validation score when the model is done fitting and proceed to plot the
    training and validation loss and accuracy curves.
    
    Input:  3 keras objects
    Output: Seaborn plots(matplotlib objects), keras weight checkpoints, trained CNN model
    """
    
    import json
    from tensorflow.keras import callbacks
    import pandas as pd
    import os
    from plot_helper import acc_plotter, loss_plotter
    import sys
    sys.path.append("..")

    checkpoint_path = '../model_checkpoints/'
    model_path = '../saved_models/'
    
    paths = [checkpoint_path, model_path]
    
    for path in paths:
        if not os.path.exists(path):
                os.mkdir(path)
    
    filepath= checkpoint_path + "little_model_weights_improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"            
    
    my_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta = 0.002, 
            mode= 'min', 
            patience=10),
        
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=4, 
            verbose=0,
            min_delta=0.01, 
            cooldown=1, 
            min_lr=0, 
            mode = 'min'),
        
        callbacks.ModelCheckpoint(
            filepath, 
            monitor='val_accuracy', 
            verbose=0, 
            save_best_only=True, 
            mode='max', 
            save_weights_only = True)
        ]
    
    print('Fitting the model.')
    
    history = model.fit(
        my_gen(train, epoch, steps_per_epoch),
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        validation_data=val,
        callbacks=my_callbacks
    )
    
    val.reset()
    
    _, val_acc = model.evaluate(val, verbose=0)
    
    print('Ftting complete with validation accuracy: {val_acc}. Now saving the fitted model.')
    
    print('Ftting complete. Now saving the fitted model')
    
    # Saving the trained keras model
    model.save('../saved_models/little_model.h5')
    
    # Saving the trained model's losses and accuracies as a csv file
    pd.DataFrame(history.history).to_csv("../data/model_history.csv")
    
    print("Saving complete. Now plotting the model's losses and accuracies.")
    
    loss_plotter(history)
    acc_plotter(history)


if __name__ == '__main__':
    import 3_data_generator as data_generator
    dim = 150
    train_gen, val_gen, _, _ = data_generator.data_gen(dim)
    model = personal_model(dim = dim)
    model_fit(train_gen, val_gen, model)