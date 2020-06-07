import data_generator

def my_model(dim = 150, lr = 0.005):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

    img_width, img_height = dim, dim
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(input_shape)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), input_shape=(input_shape)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))


    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12))
    model.add(Activation('softmax'))

    optimizer = SGD(lr=lr)

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    
    print(model.summary())
    return model

def my_gen(gen):
    steps_per_epoch = 100
    epoch  = 50
    i=0
    while i < steps_per_epoch * epoch:
        try:
            data, labels = next(gen)
            i+=1
            yield data, labels
        except:
            pass
        
def model_fit(train, val, model, epoch=50, steps_per_epoch = 100):
    from tensorflow.keras import callbacks

    checkpoint_path = '../model_checkpoints/'
    model_path = '../saved_models/'
    
    paths = [checkpoint_path, model_path]
    
    for path in paths:
        if not os.path.exists(path):
                os.mkdir(path)
    
    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta = 0.002, 
                                mode= 'min', 
                                patience=10),
        
        callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.5, 
                                    patience=4, 
                                    verbose=0,
                                    min_delta=0.01, 
                                    cooldown=1, 
                                    min_lr=0, 
                                    mode = 'min'),
        
        callbacks.ModelCheckpoint(checkpoint_path, 
                                  monitor='val_accuracy', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max', 
                                  save_weights_only = True)
        ]
    
    print('Fitting the model')
    
    history = model.fit(
        my_gen(train),
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        validation_data=val,
        callbacks=my_callbacks
        )
    
    model.save('../saved_models/my_model.h5')

    return history

if __name__ == '__main__':
    dim = 150
    train_gen, val_gen, _, _ = data_generator.data_gen(dim)
    model = my_model(dim = dim)
    model_fit(train_gen, val_gen, model)