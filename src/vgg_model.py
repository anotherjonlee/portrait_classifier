if __name__ == '__main__':
    train_gen, val_gen, _, _ = data_generator.data_gen()
    model = my_model()
    model_fit(train_gen, val_gen, model)