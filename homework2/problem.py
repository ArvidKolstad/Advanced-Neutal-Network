import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import pandas as pd
import utils as u
import pickle
from torchinfo import summary


def mlp_problem1():
    resolution = 64
    layers = [resolution * resolution, 3, 3, 1]

    act_funcs = ["ReLU", "ReLU", "sigmoid"]
    epochs = 15
    learning_rate = 0.001

    loss_function = nn.BCELoss()

    file_name = "mlp_easy_data_set"

    data = u.ImageDictDataset(pd.read_pickle("./simple_particle_dataset.pkl"))
    u.plot_images(data, "./figures/problem1/plot_training_set.png")

    data_particle_count = u.TrainingParticleCount(data)
    train, test = torch.utils.data.random_split(data_particle_count, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True, num_workers=15
    )

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=15
    )

    mlp = u.MultilayerPerception(layers, act_funcs, dropout_rate=dropout_rate)
    print(mlp)

    optimizer = torch.optim.RMSprop(mlp.parameters(), lr=learning_rate)
    u.run_training(
        mlp, epochs, loss_function, optimizer, train_loader, test_loader, file_name
    )

    trained_model = u.MultilayerPerception(layers, act_funcs)
    trained_model.load_state_dict(torch.load("./saved_models/mlp_easy_data_set"))
    trained_model.to(torch.device("cuda"))

    u.plot_roc_binary(
        trained_model,
        test_loader,
        loss_function,
        "./figures/problem1/roc_curve_mlp.png",
    )
    summary(trained_model, input_size=(32, 1, 64, 64))


def conv_problem1():
    conv_layout = [1, 2]
    epochs = 10
    loss_function = nn.BCELoss()
    mlp_layout = [2 * 2**2, 1]

    activation_functions = ["sigmoid"]

    learning_rate = 0.001

    file_name = "conv_easy_data_set"

    data = u.ImageDictDataset(pd.read_pickle("./simple_particle_dataset.pkl"))

    data_particle_count_2d = u.TrainingParticleCount(data, flat=False)

    train_2d, test_2d = torch.utils.data.random_split(
        data_particle_count_2d, [0.8, 0.2]
    )

    train_loader_2d = torch.utils.data.DataLoader(
        train_2d, batch_size=32, shuffle=True, num_workers=15
    )

    test_loader_2d = torch.utils.data.DataLoader(
        test_2d, batch_size=256, shuffle=False, num_workers=15
    )
    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout, mlp_layout, activation_functions
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    print(conv_train)
    u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader_2d,
        test_loader_2d,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout, mlp_layout, activation_functions
    )
    trained_model.load_state_dict(torch.load("./saved_models/conv_easy_data_set"))
    trained_model.to(torch.device("cuda"))

    u.plot_roc_binary(
        trained_model,
        test_loader_2d,
        loss_function,
        "./figures/problem1/roc_curve_conv.png",
    )

    summary(trained_model, input_size=(32, 1, 64, 64))


def conv_problem2():
    conv_layout = [1, 2]
    epochs = 20
    number_of_classes = 6

    mlp_layout = [2 * 2**2, number_of_classes]
    activation_functions = ["identity"]

    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.001

    file_name = "conv_hard_data_set"

    data = u.ImageDictDataset(pd.read_pickle("./hard_particle_dataset.pkl"))

    data_particle_count_2d = u.TrainingParticleCount(data, flat=False)

    train_2d, test_2d = torch.utils.data.random_split(
        data_particle_count_2d, [0.8, 0.2]
    )

    train_loader_2d = torch.utils.data.DataLoader(
        train_2d, batch_size=32, shuffle=True, num_workers=15
    )
    test_loader_2d = torch.utils.data.DataLoader(
        test_2d, batch_size=256, shuffle=False, num_workers=15
    )
    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout, mlp_layout, activation_functions
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    print(conv_train)
    u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader_2d,
        test_loader_2d,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )
    trained_model.load_state_dict(torch.load("./saved_models/conv_hard_data_set"))
    trained_model.to(torch.device("cuda"))

    u.plot_roc_multiclass(
        trained_model,
        test_loader_2d,
        number_of_classes,
        "./figures/problem2/roc_curve_conv.png",
    )
    summary(trained_model, input_size=(32, 1, 64, 64))


def conv2_problem2():
    conv_layout = [1, 2]
    epochs = 20
    number_of_classes = 6

    mlp_layout = [2 * 2**2, number_of_classes]
    activation_functions = ["identity"]

    loss_function = nn.CrossEntropyLoss()

    learning_rate = 0.001

    file_name = "conv_hard_data_set_norm"

    data = u.ImageDictDataset(pd.read_pickle("./hard_particle_dataset.pkl"))

    data_particle_count_2d = u.TrainingParticleCount(data, flat=False)

    train_2d, test_2d = torch.utils.data.random_split(
        data_particle_count_2d, [0.8, 0.2]
    )

    train_loader_2d = torch.utils.data.DataLoader(
        train_2d, batch_size=32, shuffle=True, num_workers=15
    )

    train_mean, train_std = u.get_mean_and_std_input(train_loader_2d)
    norm_transform = transforms.Normalize(mean=[train_mean], std=[train_std])
    train_loader_2d.dataset.transform = norm_transform

    test_loader_2d = torch.utils.data.DataLoader(
        test_2d, batch_size=256, shuffle=False, num_workers=15
    )
    test_loader_2d.dataset.transform = norm_transform

    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    print(conv_train)
    u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader_2d,
        test_loader_2d,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )
    trained_model.load_state_dict(torch.load("./saved_models/" + file_name))
    trained_model.to(torch.device("cuda"))

    u.plot_roc_multiclass(
        trained_model,
        test_loader_2d,
        number_of_classes,
        "./figures/problem2/roc_curve_conv_norm.png",
    )

    summary(trained_model, input_size=(32, 1, 64, 64))


def conv3_problem2():
    conv_layout = [1, 16, 32, 64]
    epochs = 40
    number_of_classes = 6
    mlp_layout = [64 * 2**2, 128, number_of_classes]
    activation_functions = ["ReLU", "identity"]
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.0005
    dropout_rate = 0.4

    file_name = "conv_hard_data_best"

    data = u.ImageDictDataset(pd.read_pickle("./hard_particle_dataset.pkl"))

    data_particle_count_2d = u.TrainingParticleCount(data, flat=False)

    train_2d, test_2d = torch.utils.data.random_split(
        data_particle_count_2d, [0.8, 0.2]
    )

    train_loader_2d = torch.utils.data.DataLoader(
        train_2d, batch_size=32, shuffle=True, num_workers=15
    )

    train_mean, train_std = u.get_mean_and_std_input(train_loader_2d)
    norm_transform = transforms.Normalize(mean=[train_mean], std=[train_std])

    test_loader_2d = torch.utils.data.DataLoader(
        test_2d, batch_size=256, shuffle=False, num_workers=15
    )

    aug_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            norm_transform,
        ]
    )
    train_loader_2d.dataset.transform = aug_transform
    test_loader_2d.dataset.transform = norm_transform

    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout, mlp_layout, activation_functions, dropout_rate=dropout_rate
    )

    optimizer = torch.optim.Adam(
        conv_train.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    print(conv_train)
    u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader_2d,
        test_loader_2d,
        file_name,
        scheduler=scheduler,
    )

    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout, mlp_layout, activation_functions, dropout_rate=dropout_rate
    )

    trained_model.load_state_dict(torch.load("./saved_models/" + file_name))
    trained_model.to(torch.device("cuda"))

    u.plot_roc_multiclass(
        trained_model,
        test_loader_2d,
        number_of_classes,
        "./figures/problem2/roc_curve_conv_best.png",
    )

    summary(trained_model, input_size=(32, 1, 64, 64))


def prune_data():

    dict_raw_data = pd.read_pickle("./simple_particle_dataset.pkl")
    raw_data = pd.DataFrame.from_dict(dict_raw_data)
    pruned_data = raw_data[
        raw_data["labels"].apply(lambda x: len(x) > 0 and x[0] > 0.0)
    ].copy()

    dict_pruned = {
        "images": np.stack(pruned_data["images"].values),
        "labels": np.stack(pruned_data["labels"].values),
    }

    data = u.ImageDictDataset(dict_pruned)

    data_particle_count_2d = u.TrainingParticlePosition(data, flat=False)
    with open("pruned_easy_set.pickle", "wb") as handle:
        pickle.dump(dict_pruned, handle)


def conv_problem3():
    conv_layout = [1, 2, 4, 8]
    epochs = 100
    dim = 2
    mlp_layout = [8 * 2**2, dim]
    activation_functions = ["identity"]
    loss_function = nn.L1Loss()
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "conv_position_easy"

    data = u.ImageDictDataset(pd.read_pickle("./pruned_easy_set.pickle"))

    data_particle_count = u.TrainingParticlePosition(data, flat=False)

    train, test = torch.utils.data.random_split(data_particle_count, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True, num_workers=15
    )

    train_mean, train_std = u.get_mean_and_std_input(train_loader)
    norm_transform = transforms.Normalize(mean=[train_mean], std=[train_std])
    train_loader.dataset.transform = norm_transform

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=15
    )
    test_loader.dataset.transform = norm_transform

    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    # print(conv_train)
    val_loss = u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader,
        test_loader,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )
    trained_model.load_state_dict(torch.load("./saved_models/" + file_name))
    trained_model.to(torch.device("cuda"))

    test_batch = next(iter(test_loader))
    u.plot_predictions(
        trained_model, test_batch, "./figures/problem3/conv_model1", device, val_loss
    )
    summary(trained_model, input_size=(32, 1, 64, 64))


def conv2_problem3():
    conv_layout = [1, 1, 1, 1, 2]
    epochs = 1000
    dim = 2
    mlp_layout = [2 * 2**2, dim]
    activation_functions = ["identity"]
    loss_function = nn.L1Loss()
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "conv2_position_easy"

    data = u.ImageDictDataset(pd.read_pickle("./pruned_easy_set.pickle"))

    data_particle_count = u.TrainingParticlePosition(data, flat=False)

    train, test = torch.utils.data.random_split(data_particle_count, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True, num_workers=15
    )

    train_mean, train_std = u.get_mean_and_std_input(train_loader)
    norm_transform = transforms.Normalize(mean=[train_mean], std=[train_std])
    train_loader.dataset.transform = norm_transform

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=15
    )
    test_loader.dataset.transform = norm_transform

    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    # print(conv_train)
    val_loss = u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader,
        test_loader,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )
    trained_model.load_state_dict(torch.load("./saved_models/" + file_name))
    trained_model.to(torch.device("cuda"))

    test_batch = next(iter(test_loader))
    u.plot_predictions(
        trained_model, test_batch, "./figures/problem3/conv_model2", device, val_loss
    )
    summary(trained_model, input_size=(32, 1, 64, 64))


def conv3_problem3():
    conv_layout = [1, 16, 16]
    epochs = 1000
    dim = 2
    mlp_layout = [16 * 2**2, dim]
    activation_functions = ["identity"]
    loss_function = nn.L1Loss()
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "conv3_position_easy"

    data = u.ImageDictDataset(pd.read_pickle("./pruned_easy_set.pickle"))

    data_particle_count = u.TrainingParticlePosition(data, flat=False)

    train, test = torch.utils.data.random_split(data_particle_count, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True, num_workers=15
    )

    train_mean, train_std = u.get_mean_and_std_input(train_loader)
    norm_transform = transforms.Normalize(mean=[train_mean], std=[train_std])
    train_loader.dataset.transform = norm_transform

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=15
    )
    test_loader.dataset.transform = norm_transform

    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    # print(conv_train)
    val_loss = u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader,
        test_loader,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )
    trained_model.load_state_dict(torch.load("./saved_models/" + file_name))
    trained_model.to(torch.device("cuda"))

    test_batch = next(iter(test_loader))
    u.plot_predictions(
        trained_model, test_batch, "./figures/problem3/conv_model3", device, val_loss
    )
    summary(trained_model, input_size=(32, 1, 64, 64))


def conv1_problem4():
    conv_layout = [1, 1, 1, 1, 2]
    epochs = 1000
    dim = 2
    mlp_layout = [2 * 2**2, dim]
    activation_functions = ["identity"]
    loss_function = nn.L1Loss()
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "conv1_position_hard"

    data = u.ImageDictDataset(pd.read_pickle("./pruned_hard_set.pickle"))

    data_particle_count = u.TrainingParticlePosition(data, flat=False)

    train, test = torch.utils.data.random_split(data_particle_count, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True, num_workers=15
    )

    train_mean, train_std = u.get_mean_and_std_input(train_loader)
    norm_transform = transforms.Normalize(mean=[train_mean], std=[train_std])
    train_loader.dataset.transform = norm_transform

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=15
    )
    test_loader.dataset.transform = norm_transform

    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    # print(conv_train)
    val_loss = u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader,
        test_loader,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )
    trained_model.load_state_dict(torch.load("./saved_models/" + file_name))
    trained_model.to(torch.device("cuda"))

    test_batch = next(iter(test_loader))
    u.plot_predictions(
        trained_model, test_batch, "./figures/problem4/conv_model1", device, val_loss
    )
    summary(trained_model, input_size=(32, 1, 64, 64))


def conv2_problem4():
    conv_layout = [1, 2, 4, 8, 16]
    epochs = 1000
    dim = 2
    mlp_layout = [16 * 2**2, dim]
    activation_functions = ["identity"]
    loss_function = nn.L1Loss()
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "conv2_position_hard"

    data = u.ImageDictDataset(pd.read_pickle("./pruned_hard_set.pickle"))

    data_particle_count = u.TrainingParticlePosition(data, flat=False)

    train, test = torch.utils.data.random_split(data_particle_count, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True, num_workers=15
    )

    train_mean, train_std = u.get_mean_and_std_input(train_loader)
    norm_transform = transforms.Normalize(mean=[train_mean], std=[train_std])
    train_loader.dataset.transform = norm_transform

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=15
    )
    test_loader.dataset.transform = norm_transform

    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    # print(conv_train)
    val_loss = u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader,
        test_loader,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )
    trained_model.load_state_dict(torch.load("./saved_models/" + file_name))
    trained_model.to(torch.device("cuda"))

    test_batch = next(iter(test_loader))
    u.plot_predictions(
        trained_model, test_batch, "./figures/problem4/conv_model2", device, val_loss
    )
    summary(trained_model, input_size=(32, 1, 64, 64))


def conv3_problem4():
    conv_layout = [1, 1, 1, 1, 1]
    epochs = 1000
    dim = 2
    mlp_layout = [1 * 2**2, 8, 8, dim]
    activation_functions = ["ReLU", "ReLU", "identity"]
    loss_function = nn.L1Loss()
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "conv3_position_hard"

    data = u.ImageDictDataset(pd.read_pickle("./pruned_hard_set.pickle"))

    data_particle_count = u.TrainingParticlePosition(data, flat=False)

    train, test = torch.utils.data.random_split(data_particle_count, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=32, shuffle=True, num_workers=15
    )

    train_mean, train_std = u.get_mean_and_std_input(train_loader)
    norm_transform = transforms.Normalize(mean=[train_mean], std=[train_std])
    train_loader.dataset.transform = norm_transform

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256, shuffle=False, num_workers=15
    )
    test_loader.dataset.transform = norm_transform

    conv_train = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )

    optimizer = torch.optim.RMSprop(conv_train.parameters(), lr=learning_rate)

    # print(conv_train)
    val_loss = u.run_training(
        conv_train,
        epochs,
        loss_function,
        optimizer,
        train_loader,
        test_loader,
        file_name,
    )
    trained_model = u.ConvolutionalNeuralNetwork(
        conv_layout,
        mlp_layout,
        activation_functions,
    )
    trained_model.load_state_dict(torch.load("./saved_models/" + file_name))
    trained_model.to(torch.device("cuda"))

    test_batch = next(iter(test_loader))
    u.plot_predictions(
        trained_model, test_batch, "./figures/problem4/conv_model3", device, val_loss
    )
    summary(trained_model, input_size=(32, 1, 64, 64))


def main():
    # mlp_problem1()
    # conv_problem1()
    # conv_problem2()
    # conv2_problem2()
    conv3_problem2()
    # prune_data()
    # conv_problem3()
    # conv2_problem3()
    # conv3_problem3()
    # conv1_problem4()
    # conv2_problem4()
    # conv3_problem4()


if __name__ == "__main__":
    main()
