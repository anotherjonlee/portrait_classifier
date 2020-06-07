def evaluate_model(model_path, val_gen, test_gen):
    from tensorflow.keras.models import Model, load_model
    import sys
    sys.path.append("..") ## resetting the path to the parent directory



if __name__ == '__main__':
    dim = 150
    _, _, evaluate_gen, test_gen = data_generator.data_gen(dim)
    model = my_model(dim = dim)
    model_fit(train_gen, val_gen, model)