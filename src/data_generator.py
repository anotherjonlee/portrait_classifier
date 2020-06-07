
def data_gen(dim = 224):
    
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

    batch_size = 32

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        '../data/train_folder',  # this is the target directory
        target_size=(dim, dim),  # all images will be resized to 150x150
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
    
    validation_generator = val_datagen.flow_from_directory(
        '../data/validation_folder',
        target_size=(dim, dim),
        batch_size=batch_size,
        class_mode='categorical')
    
    evaluate_generator = eval_datagen.flow_from_directory(
        '../data/validation_folder',
        target_size=(dim, dim),
        batch_size=1,
        seed=42,
        shuffle=False,
        class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(
        directory='../data/holdout_folder',
        target_size=(dim, dim),
        color_mode="rgb",
        batch_size=1,
        shuffle=False,
        class_mode=None,
        seed=42)

    return train_generator, validation_generator, evaluate_generator, test_generator