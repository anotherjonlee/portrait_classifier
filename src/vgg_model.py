
def my_model(dim, lr = 0.001):
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
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    import numpy as np
    import pandas as pd
    import data_generator
    
    img_width, img_height = dim, dim
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    
    else:
        input_shape = (img_width, img_height, 3)
    
    base_model = VGG16(
        include_top=False, 
        weights='imagenet',
        input_shape = (dim,dim,3), 
        pooling=max)

    print(f'Base VGG-16 architecture: {base_model.summary()}')
    
    # Unfreezing the last convolution and pooling layer to train it with coin images
    base_model.trainable = True

    set_trainable = False
    
    for layer in base_model.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    
    # Adding my own dense layer at the end of the VGG-16 model
    
    inpt = Input(
        shape=input_shape,
        name = 'image_input')
    
    output = base_model(inpt)
    
    flat1 = Flatten(name='flatten')(output)
    class1 = Dense(1024, activation='relu', name='fc1')(flat1)
    output = Dense(12, activation='softmax', name='predictions')(class1)
    
    # Define a modified model
    vgg_model = Model(inputs=inpt, outputs=output)
    
    print(f'Modified VGG-16 model architecture: {vgg_model.summary()}')
    
    optimizer = SGD(lr=0.001,momentum=0.9)
    
    vgg_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    

def my_gen(gen, epoch, steps_per_epoch):
    
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
    
    from tensorflow.keras import callbacks
    from plot_helper import acc_plotter, loss_plotter
    import sys
    sys.path.append("..")
    
    checkpoint_path = '../model_checkpoints/'
    model_path = '../saved_models/'
    
    paths = [checkpoint_path, model_path]
    
    for path in paths:
        if not os.path.exists(path):
                os.mkdir(path)
                
    filepath= checkpoint_path + "vgg_v3_partial_unfreeze_weights_improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"            
    
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
            save_weights_only = True,
            mode='max')
        ]   

    print('Fitting the model.')
    
    history = vgg_model.fit(
        my_gen(train, epoch, steps_per_epoch),
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        validation_data=val,
        callbacks=my_callbacks
        )
    
    val.reset()
    
    _, val_acc = vgg_model.evaluate(val, verbose=0)
    
    print('Ftting complete with validation accuracy: {val_acc}. Now saving the fitted model.')
    
    vgg_model.save('../saved_models/vgg_model.h5')    
    
    print("Saving complete. Now plotting the model's losses and accuracies.")
    
    loss_plotter(history)
    acc_plotter(history)

if __name__ == '__main__':
    dim = 224
    train_gen, val_gen, _, _ = data_generator.data_gen()
    model = my_model(dim = dim)
    model_fit(train_gen, val_gen, model)