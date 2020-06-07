
def evaluator(model_name):
    """
    The function takes a name of the trained VGG-16 model that's been trained from either vgg_model.py file.
    It extracts the true y values and the predicted y values to print out the precision, recall and f1 scores.
    Additionally, the function generates a confusion matrix.
    
    Input:  Name of the fitted and saved model (string)
    Output: Confusion matrix (matplotlib object) and the precision, recall and f1-score report
    """
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.models import load_model
    from sklearn.metrics import classification_report
    from data_generator import data_gen
    from plot_helper import cm_plotter
    import sys
    sys.path.append("..")
    
    if model_name[-2:] != 'h5'
        model_path = '../saved_models/' + model_name + '.h5'
    else:
        model_path = '../saved_models/' + model_name

    # Load the saved model
    model = load_model(model_path)

    _, _, test_generator = data_gen()
    
    pred = model.predict(
        test_generator, 
        verbose = 0
    )

    y_pred = np.argmax(
        pred,
        axis = 1
    )
    
    y_true = test_generator.classes
    
    class_labels = test_generator.class_indices.keys()
    
    report = classification_report(
        y_true
        y_pred,
        target_names = class_labels
    )
    
    print(f"Printing model's Precision, Recall and f1 scores.")
    print(f'{report}')
    print('Plotting the confusion matrix.')
    cm_plotter(y_true, y_pred, class_labels)

if __name__ == '__main__':
    model_name = 'vgg_model'
    evaluator(model_name)