import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Single_layer_network:
    def __init__(self, input_size, output_size, activation_func):
        self.x_size = input_size
        self.y_size = output_size

        self.act_func = self.get_activation_func(activation_func)
        self.act_func_derivative = self.get_activation_derivative(activation_func)

        self.w = np.random.normal(size=(self.y_size, self.x_size))
        self.b = np.random.normal(size=(self.y_size, 1))

    def b_i(self, x):
        b_i = self.w @ x + self.b
        return b_i

    def forward_feed(self, x):
        y_pred = self.act_func(self.b_i(x))
        return y_pred

    def error_function(self, y_pred, y_true):
        return (y_true - y_pred) ** 2

    def forward_feed_array(self, x_input):
        y_pred = np.zeros((x_input.shape[0], self.y_size))
        for i in range(x_input.shape[0]):
            y_pred[i] = self.forward_feed(x_input[i])
        return y_pred

    def get_activation_func(self, act_func: str):
        if act_func == "sigmoid":
            func = sigmoid
        elif act_func == "tanh":
            func = np.tanh
        else:
            raise ValueError("activation function is not implemented")
        return func

    def get_activation_derivative(self, act_func):
        if act_func == "sigmoid":
            func = sigmoid_derivative
        elif act_func == "tanh":
            func = tanh_derivative

        else:
            raise ValueError("activation function is not implemented")
        return func

    def back_propagation(self, x_input: np.ndarray, y_target: np.ndarray):
        y_pred = self.forward_feed(x_input)
        delta_w = (
            -2
            * (y_target - y_pred)
            * self.act_func_derivative(self.b_i(x_input))
            * x_input
        )
        delta_b = -2 * (y_target - y_pred) * self.act_func_derivative(self.b_i(x_input))
        return delta_w, delta_b

    def training_loop(
        self,
        training_iterations,
        learning_rate,
        x_input: np.ndarray,
        y_target: np.ndarray,
    ):
        data_length = x_input.shape[0]
        errors = np.zeros((training_iterations, 1))
        for i in range(training_iterations):
            rand_index = np.random.randint(0, data_length)
            delta_w, delta_b = self.back_propagation(
                x_input[rand_index], y_target[rand_index]
            )
            y_pred = self.forward_feed(x_input[rand_index])
            errors[i] = self.error_function(y_pred, y_target[rand_index])
            self.w -= delta_w * learning_rate
            self.b -= delta_b * learning_rate
        return errors

    def batch_training_loop(
        self, training_iterations, batch_size, learning_rate, x_input, y_target
    ):
        data_length = x_input.shape[0]
        errors = np.zeros((training_iterations, self.y_size))
        for i in range(training_iterations):
            idx = np.random.choice(data_length, size=batch_size, replace=False)
            x_batch = x_input[idx].reshape((batch_size, self.x_size))
            y_batch = y_target[idx].reshape((batch_size, self.y_size))

            delta_w = 0
            delta_b = 0
            error = 0
            for j in range(batch_size):
                new_delta_w, new_delta_b = self.back_propagation(x_batch[j], y_batch[j])
                delta_w += new_delta_w
                delta_b += new_delta_b
                y_pred = self.forward_feed(x_batch[j])
                error += self.error_function(y_pred, y_batch[j])
            errors[i] = error / batch_size
            self.w -= delta_w / batch_size * learning_rate
            self.b -= delta_b / batch_size * learning_rate

        return errors


class Double_layer_network:
    def __init__(self, input_size, hidden_size, output_size, activation_func):
        self.x_size = input_size
        self.h_size = hidden_size
        self.y_size = output_size

        self.act_func = self.get_activation_func(activation_func)
        self.act_func_derivative = self.get_activation_derivative(activation_func)

        self.w_in = np.random.normal(size=(self.h_size, self.x_size))
        self.b_in = np.random.normal(size=(self.h_size, 1))

        self.w_hidden = np.random.normal(size=(self.y_size, self.h_size))
        self.b_hidden = np.random.normal(size=(self.y_size, 1))

    def b_i(self, x, layer) -> np.ndarray:
        if layer == "hidden":
            x_reshape = np.reshape(x, (-1, 1))
            b_i = self.w_in @ x_reshape + self.b_in
        elif layer == "output":
            x_reshape = np.reshape(x, (-1, 1))
            b_i = self.w_hidden @ x_reshape + self.b_hidden
        else:
            raise ValueError("Layer is invalid")
        return b_i

    def forward_feed(self, x: np.ndarray) -> tuple:
        b_hidden = self.b_i(x, "hidden")
        s_hidden = self.act_func(b_hidden)
        b_output = self.b_i(s_hidden, "output")
        y_pred = self.act_func(b_output)
        node_values = (y_pred, b_output, s_hidden, b_hidden)
        return node_values

    def error_function(self, y_pred, y_true):
        return (y_true - y_pred) ** 2

    def forward_feed_array(self, x_input):
        y_pred = np.zeros((x_input.shape[0], self.y_size))
        for i in range(x_input.shape[0]):
            y_pred[i] = self.forward_feed(x_input[i])[0]
        return y_pred

    def get_activation_func(self, act_func: str):
        if act_func == "sigmoid":
            func = sigmoid
        elif act_func == "tanh":
            func = np.tanh
        else:
            raise ValueError("activation function is not implemented")
        return func

    def get_activation_derivative(self, act_func):
        if act_func == "sigmoid":
            func = sigmoid_derivative
        elif act_func == "tanh":
            func = tanh_derivative

        else:
            raise ValueError("activation function is not implemented")
        return func

    def back_propagation(self, x_input: np.ndarray, y_target: np.ndarray):
        x = np.reshape(x_input, (-1, 1))
        y_target = np.reshape(y_target, (-1, 1))

        y_pred, b_output, s_hidden, b_hidden = self.forward_feed(x)

        delta_b_hidden = -2 * (y_target - y_pred) * self.act_func_derivative(b_output)

        delta_w_hidden = delta_b_hidden @ s_hidden.T

        delta_b_in = (self.w_hidden.T @ delta_b_hidden) * self.act_func_derivative(
            b_hidden
        )

        delta_w_in = delta_b_in @ x.T

        return delta_w_hidden, delta_b_hidden, delta_w_in, delta_b_in

    def training_loop(
        self,
        training_iterations,
        learning_rate,
        x_input: np.ndarray,
        y_target: np.ndarray,
    ):
        data_length = x_input.shape[0]
        errors = np.zeros((training_iterations, 1))
        for i in range(training_iterations):
            rand_index = np.random.randint(0, data_length)
            delta_w_hidden, delta_b_hidden, delta_w_in, delta_b_in = (
                self.back_propagation(x_input[rand_index], y_target[rand_index])
            )
            y_out = self.forward_feed(x_input[rand_index])[0]
            errors[i] = self.error_function(y_out, y_target[rand_index])

            self.w_in -= delta_w_in * learning_rate
            self.b_in -= delta_b_in * learning_rate
            self.w_hidden -= delta_w_hidden * learning_rate
            self.b_hidden -= delta_b_hidden * learning_rate
        return errors

    def batch_training_loop(
        self, training_iterations, batch_size, learning_rate, x_input, y_target
    ):
        data_length = x_input.shape[0]
        errors = np.zeros((training_iterations, self.y_size))
        for i in range(training_iterations):
            idx = np.random.choice(data_length, size=batch_size, replace=False)
            x_batch = x_input[idx].reshape((batch_size, self.x_size))
            y_batch = y_target[idx].reshape((batch_size, self.y_size))

            delta_w_in = np.zeros_like(self.w_in)
            delta_b_in = np.zeros_like(self.b_in)
            delta_w_hidden = np.zeros_like(self.w_hidden)
            delta_b_hidden = np.zeros_like(self.b_hidden)

            error = 0
            for j in range(batch_size):
                (
                    new_delta_w_hidden,
                    new_delta_b_hidden,
                    new_delta_w_in,
                    new_delta_b_in,
                ) = self.back_propagation(x_batch[j], y_batch[j])

                delta_w_in += new_delta_w_in
                delta_b_in += new_delta_b_in
                delta_w_hidden += new_delta_w_hidden
                delta_b_hidden += new_delta_b_hidden

                y_out = self.forward_feed(x_batch[j])[0]
                error += self.error_function(y_out, y_batch[j])
            errors[i] = error / batch_size

            self.w_in -= delta_w_in / batch_size * learning_rate
            self.b_in -= delta_b_in / batch_size * learning_rate
            self.w_hidden -= delta_w_hidden / batch_size * learning_rate
            self.b_hidden -= delta_b_hidden / batch_size * learning_rate

        return errors


class Triple_layer_network:
    def __init__(
        self, input_size, hidden_size1, hidden_size2, output_size, activation_func
    ):
        self.x_size = input_size
        self.h1_size = hidden_size1
        self.h2_size = hidden_size2
        self.y_size = output_size

        self.act_func = self.get_activation_func(activation_func)
        self.act_func_derivative = self.get_activation_derivative(activation_func)

        self.w_1 = np.random.normal(size=(self.h1_size, self.x_size))
        self.b_1 = np.random.normal(size=(self.h1_size, 1))

        self.w_2 = np.random.normal(size=(self.h2_size, self.h1_size))
        self.b_2 = np.random.normal(size=(self.h2_size, 1))

        self.w_3 = np.random.normal(size=(self.y_size, self.h2_size))
        self.b_3 = np.random.normal(size=(self.y_size, 1))

    def b_i(self, x, layer) -> np.ndarray:
        if layer == "1":
            x_reshape = np.reshape(x, (-1, 1))
            b_i = self.w_1 @ x_reshape + self.b_1
        elif layer == "2":
            x_reshape = np.reshape(x, (-1, 1))
            b_i = self.w_2 @ x_reshape + self.b_2
        elif layer == "3":
            x_reshape = np.reshape(x, (-1, 1))
            b_i = self.w_3 @ x_reshape + self.b_3
        else:
            raise ValueError("Layer is invalid")
        return b_i

    def forward_feed(self, x: np.ndarray) -> tuple:
        b_1 = self.b_i(x, "1")
        s_1 = self.act_func(b_1)
        b_2 = self.b_i(s_1, "2")
        s_2 = self.act_func(b_2)
        b_3 = self.b_i(s_2, "3")
        s_3 = self.act_func(b_3)
        node_values = (s_3, b_3, s_2, b_2, s_1, b_1)
        return node_values

    def error_function(self, y_pred, y_true):
        return (y_true - y_pred) ** 2

    def forward_feed_array(self, x_input):
        y_pred = np.zeros((x_input.shape[0], self.y_size))
        for i in range(x_input.shape[0]):
            y_pred[i] = self.forward_feed(x_input[i])[0]
        return y_pred

    def get_activation_func(self, act_func: str):
        if act_func == "sigmoid":
            func = sigmoid
        elif act_func == "tanh":
            func = np.tanh
        else:
            raise ValueError("activation function is not implemented")
        return func

    def get_activation_derivative(self, act_func):
        if act_func == "sigmoid":
            func = sigmoid_derivative
        elif act_func == "tanh":
            func = tanh_derivative

        else:
            raise ValueError("activation function is not implemented")
        return func

    def back_propagation(self, x_input: np.ndarray, y_target: np.ndarray):
        x = np.reshape(x_input, (-1, 1))
        y_target = np.reshape(y_target, (-1, 1))

        s_3, b_3, s_2, b_2, s_1, b_1 = self.forward_feed(x)

        delta_b_3 = -2 * (y_target - s_3) * self.act_func_derivative(b_3)

        delta_w_3 = delta_b_3 @ s_2.T

        delta_b_2 = (self.w_3.T @ delta_b_3) * self.act_func_derivative(b_2)

        delta_w_2 = delta_b_2 @ s_1.T

        delta_b_1 = (self.w_2.T @ delta_b_2) * self.act_func_derivative(b_1)

        delta_w_1 = delta_b_1 @ x.T

        return delta_w_3, delta_b_3, delta_w_2, delta_b_2, delta_w_1, delta_b_1

    def training_loop(
        self,
        training_iterations,
        learning_rate,
        x_input: np.ndarray,
        y_target: np.ndarray,
    ):
        data_length = x_input.shape[0]
        errors = np.zeros((training_iterations, 1))
        for i in range(training_iterations):
            rand_index = np.random.randint(0, data_length)
            delta_w_3, delta_b_3, delta_w_2, delta_b_2, delta_w_1, delta_b_1 = (
                self.back_propagation(x_input[rand_index], y_target[rand_index])
            )
            y_out = self.forward_feed(x_input[rand_index])[0]
            errors[i] = self.error_function(y_out, y_target[rand_index])

            self.w_1 -= delta_w_1 * learning_rate
            self.b_1 -= delta_b_1 * learning_rate
            self.w_2 -= delta_w_2 * learning_rate
            self.b_2 -= delta_b_2 * learning_rate
            self.w_3 -= delta_w_3 * learning_rate
            self.b_3 -= delta_b_3 * learning_rate
        return errors

    def batch_training_loop(
        self, training_iterations, batch_size, learning_rate, x_input, y_target
    ):
        data_length = x_input.shape[0]
        errors = np.zeros((training_iterations, self.y_size))
        for i in range(training_iterations):
            idx = np.random.choice(data_length, size=batch_size, replace=False)
            x_batch = x_input[idx].reshape((batch_size, self.x_size))
            y_batch = y_target[idx].reshape((batch_size, self.y_size))

            delta_w_3 = np.zeros_like(self.w_3)
            delta_b_3 = np.zeros_like(self.b_3)
            delta_w_2 = np.zeros_like(self.w_2)
            delta_b_2 = np.zeros_like(self.b_2)
            delta_w_1 = np.zeros_like(self.w_1)
            delta_b_1 = np.zeros_like(self.b_1)

            error = 0
            for j in range(batch_size):
                (
                    new_delta_w_3,
                    new_delta_b_3,
                    new_delta_w_2,
                    new_delta_b_2,
                    new_delta_w_1,
                    new_delta_b_1,
                ) = self.back_propagation(x_batch[j], y_batch[j])

                delta_w_1 += new_delta_w_1
                delta_b_1 += new_delta_b_1
                delta_w_2 += new_delta_w_2
                delta_b_2 += new_delta_b_2
                delta_w_3 += new_delta_w_3
                delta_b_3 += new_delta_b_3

                y_out = self.forward_feed(x_batch[j])[0]
                error += self.error_function(y_out, y_batch[j])
            errors[i] = error / batch_size

            self.w_3 -= delta_w_3 / batch_size * learning_rate
            self.b_3 -= delta_b_3 / batch_size * learning_rate
            self.w_2 -= delta_w_2 / batch_size * learning_rate
            self.b_2 -= delta_b_2 / batch_size * learning_rate
            self.w_1 -= delta_w_1 / batch_size * learning_rate
            self.b_1 -= delta_b_1 / batch_size * learning_rate

        return errors


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def problem1():
    df = pd.read_csv("./1d_classification_single_neuron.csv", header=None)
    df.columns = ["x", "y"]
    # print(df.head(10))
    x_input = df["x"].to_numpy()
    y_target = df["y"].to_numpy()
    d_len = x_input.shape[0]
    x_input = x_input.reshape(d_len, 1)
    y_target = y_target.reshape(d_len, 1)

    input_size = 1
    output_size = 1
    act_func = "sigmoid"
    NN = Single_layer_network(input_size, output_size, act_func)
    y_pred = NN.forward_feed_array(x_input)

    fig, ax = plt.subplots()

    ax.scatter(x_input, y_target, label="Target Values", c="black")
    ax.scatter(x_input, y_pred, label="Predicted Values", marker="+", c="red", s=35)
    ax.text(-0.75, 0.5, f"w = {NN.w[0,0]:.3f}")
    ax.text(-0.75, 0.4, f"b = {NN.b[0,0]:.3f}")
    ax.set_title("Untrained model")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.legend()
    fig.tight_layout()
    fig.savefig("./figures/problem1/untrained_single_neuron1d.pdf")

    training_iterations = 100000
    learning_rate = 0.01

    errors = NN.training_loop(training_iterations, learning_rate, x_input, y_target)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    y_pred = NN.forward_feed_array(x_input)
    ax[0].scatter(x_input, y_target, label="Target Values", c="black")
    ax[0].scatter(x_input, y_pred, label="Predicted Values", marker="+", c="red", s=35)
    ax[0].text(-0.75, 0.5, f"w = {NN.w[0,0]:.3f}")
    ax[0].text(-0.75, 0.4, f"b = {NN.b[0,0]:.3f}")
    ax[0].set_title("Trained model")
    ax[0].set_xlabel("Input")
    ax[0].set_ylabel("Output")
    ax[0].legend()

    ax[1].plot(errors)
    ax[1].set_xlabel("Training iterations")
    ax[1].set_ylabel("Square Error")
    ax[1].set_title("Training Error")
    fig.tight_layout()
    fig.savefig("./figures/problem1/trained_single_neuron1d.pdf")

    training_iterations = 10000
    learning_rate = 0.1
    batch_size = 50

    errors = NN.batch_training_loop(
        training_iterations, batch_size, learning_rate, x_input, y_target
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    y_pred = NN.forward_feed_array(x_input)
    ax[0].scatter(x_input, y_target, label="Target Values", c="black")
    ax[0].scatter(x_input, y_pred, label="Predicted Values", marker="+", c="red", s=35)
    ax[0].text(-0.75, 0.5, f"w = {NN.w[0,0]:.3f}")
    ax[0].text(-0.75, 0.4, f"b = {NN.b[0,0]:.3f}")
    ax[0].set_title("Trained model")
    ax[0].set_xlabel("Input")
    ax[0].set_ylabel("Output")
    ax[0].legend()

    ax[1].plot(errors)

    ax[1].set_xlabel("Training iterations")
    ax[1].set_ylabel("Square Error")
    ax[1].set_title("Training Error")
    ax[1].text(-0.75, 0.5, f"Learning rate = {learning_rate:.2f}")
    fig.tight_layout()
    fig.savefig("./figures/problem1/trained_single_neuron1d_batch.pdf")


def problem2():

    df = pd.read_csv("./2d_classification_single_neuron.csv", header=None)
    df.columns = ["x1", "x2", "y"]
    df["y"].astype(int)
    # print(df.head(10))
    x_input = df[["x1", "x2"]].to_numpy()
    y_target = df["y"].to_numpy()

    class_1 = df.query("y == 0")[["x1", "x2"]].to_numpy()
    class_2 = df.query("y == 1")[["x1", "x2"]].to_numpy()
    fig, ax = plt.subplots()
    ax.scatter(class_1[:, 0], class_1[:, 1], label="y = 0")
    ax.scatter(class_2[:, 0], class_2[:, 1], label="y = 1")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.set_title("Scatter plot over the data")
    fig.savefig("./figures/problem2/plot_data.pdf")

    input_size = 2
    output_size = 1
    activation_function = "sigmoid"
    NN = Single_layer_network(input_size, output_size, activation_function)

    training_iterations = 10000
    learning_rate = 0.1
    batch_size = 50

    errors = NN.batch_training_loop(
        training_iterations, batch_size, learning_rate, x_input, y_target
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    y_pred = NN.forward_feed_array(x_input)
    class_1_mask_pred = y_pred < 0.5
    class_2_mask_pred = y_pred > 0.5

    ax[0].scatter(class_1[:, 0], class_1[:, 1], label="y = 0")
    ax[0].scatter(class_2[:, 0], class_2[:, 1], label="y = 1")

    ax[0].scatter(
        x_input[:, 0][class_1_mask_pred[:, 0]],
        x_input[:, 1][class_1_mask_pred[:, 0]],
        label="y = 0",
        marker="+",
        s=35,
    )
    ax[0].scatter(
        x_input[:, 0][class_2_mask_pred[:, 0]],
        x_input[:, 1][class_2_mask_pred[:, 0]],
        label="y = 1",
        marker="+",
        s=35,
    )

    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Trained model")
    ax[0].legend()

    ax[1].plot(errors)

    ax[1].set_xlabel("Training iterations")
    ax[1].set_ylabel("Square Error")
    ax[1].set_title("Training Error")
    ax[1].text(-0.75, 0.5, f"Learning rate = {learning_rate:.2f}")
    fig.savefig("./figures/problem2/trained_neuron.pdf")


def problem3():
    df = pd.read_csv("./2d_classification_multiple_neurons.csv", header=None)
    df.columns = ["x1", "x2", "y"]
    df["y"].astype(int)
    # print(df.head(10))
    x_input = df[["x1", "x2"]].to_numpy()
    y_target = df["y"].to_numpy()

    class_1 = df.query("y == 0")[["x1", "x2"]].to_numpy()
    class_2 = df.query("y == 1")[["x1", "x2"]].to_numpy()
    fig, ax = plt.subplots()
    ax.scatter(class_1[:, 0], class_1[:, 1], label="y = 0")
    ax.scatter(class_2[:, 0], class_2[:, 1], label="y = 1")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.set_title("Scatter plot over the data")
    fig.savefig("./figures/problem3/plot_data.pdf")

    input_size = 2
    hidden_size = 5
    output_size = 1
    activation_func = "sigmoid"
    training_iterations = 10000
    batch_size = 50
    learning_rate = 0.1

    NN = Double_layer_network(input_size, hidden_size, output_size, activation_func)
    errors = NN.batch_training_loop(
        training_iterations, batch_size, learning_rate, x_input, y_target
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    y_pred = NN.forward_feed_array(x_input)
    class_1_mask_pred = y_pred < 0.5
    class_2_mask_pred = y_pred > 0.5

    ax[0].scatter(class_1[:, 0], class_1[:, 1], label="y = 0")
    ax[0].scatter(class_2[:, 0], class_2[:, 1], label="y = 1")

    ax[0].scatter(
        x_input[:, 0][class_1_mask_pred[:, 0]],
        x_input[:, 1][class_1_mask_pred[:, 0]],
        label="y = 0",
        marker="+",
        s=35,
    )
    ax[0].scatter(
        x_input[:, 0][class_2_mask_pred[:, 0]],
        x_input[:, 1][class_2_mask_pred[:, 0]],
        label="y = 1",
        marker="+",
        s=35,
    )

    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Trained model")
    ax[0].legend()

    ax[1].plot(errors)

    ax[1].set_xlabel("Training iterations")
    ax[1].set_ylabel("Square Error")
    ax[1].set_title("Training Error")
    ax[1].text(-0.75, 0.5, f"Learning rate = {learning_rate:.2f}")
    fig.savefig("./figures/problem3/trained_neuron.pdf")


def problem4():
    df = pd.read_csv("./function_approximation.csv", header=None)
    df.columns = ["x", "y"]
    # print(df.head(10))
    x_input = df["x"].to_numpy()
    y_target = df["y"].to_numpy()

    fig, ax = plt.subplots()
    ax.scatter(x_input, y_target)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Scatter plot over the data")
    fig.savefig("./figures/problem4/plot_data.pdf")

    input_size = 1
    hidden_size1 = 10
    hidden_size2 = 10
    output_size = 1
    activation_func = "tanh"
    training_iterations = 10000
    batch_size = 30
    learning_rate = 0.1

    NN = Triple_layer_network(
        input_size, hidden_size1, hidden_size2, output_size, activation_func
    )
    # NN = Double_layer_network(input_size, hidden_size1, output_size, activation_func)

    errors = NN.batch_training_loop(
        training_iterations, batch_size, learning_rate, x_input, y_target
    )

    function = NN.forward_feed_array(x_input)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(
        x_input,
        y_target,
        label="Target values",
    )
    ax[0].set_title("Trained model")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[0].scatter(
        x_input,
        function,
        label="Predicted",
        marker="+",
        s=35,
    )
    ax[0].legend()
    ax[1].plot(errors)
    ax[1].set_xlabel("Training iterations")
    ax[1].set_ylabel("Square Error")
    ax[1].set_title("Training Error")
    ax[1].text(-0.75, 0.5, f"Learning rate = {learning_rate:.2f}")
    fig.savefig("./figures/problem4/triple_network.pdf")


def main():
    # problem1()
    # problem2()
    # problem3()
    problem4()


if __name__ == "__main__":
    main()
