import numpy as np

SEED = 42


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def initialize_parameters(layer_sizes, activation_names, seed=SEED):
    rng = np.random.default_rng(seed)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        if activation_names[i] == "relu":
            scale = np.sqrt(2.0 / n_in)
        else:
            scale = np.sqrt(1.0 / n_in)
        W = rng.normal(0.0, scale, size=(n_in, n_out))
        b = np.zeros((1, n_out))
        weights.append(W)
        biases.append(b)
    return weights, biases


ACTIVATION_FUNCTIONS = {"relu": relu, "tanh": tanh, "softmax": softmax}


def forward_pass(X, weights, biases, activation_names):
    activations = [X]
    pre_activations = []
    current = X
    for i in range(len(weights)):
        z = np.dot(current, weights[i]) + biases[i]
        a = ACTIVATION_FUNCTIONS[activation_names[i]](z)
        pre_activations.append(z)
        activations.append(a)
        current = a
    return activations, pre_activations


def one_hot(y, n_classes):
    encoded = np.zeros((y.shape[0], n_classes))
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def cross_entropy_loss(y_true, probabilities, weights, l2_lambda):
    n_samples = y_true.shape[0]
    correct_probabilities = probabilities[np.arange(n_samples), y_true]
    data_loss = -np.mean(np.log(correct_probabilities + 1e-12))
    weight_penalty = 0.0
    for W in weights:
        weight_penalty += np.sum(W ** 2)
    return data_loss + 0.5 * l2_lambda * weight_penalty


DERIVATIVES = {"relu": relu_derivative, "tanh": tanh_derivative}


def backward_pass(y_true, activations, pre_activations, weights, activation_names, l2_lambda):
    n_samples = y_true.shape[0]
    n_layers = len(weights)
    grads_W = [None] * n_layers
    grads_b = [None] * n_layers

    y_one_hot = one_hot(y_true, activations[-1].shape[1])
    delta = (activations[-1] - y_one_hot) / n_samples

    for i in reversed(range(n_layers)):
        grads_W[i] = np.dot(activations[i].T, delta) + l2_lambda * weights[i]
        grads_b[i] = np.sum(delta, axis=0, keepdims=True)
        if i > 0:
            delta_previous = np.dot(delta, weights[i].T)
            previous_name = activation_names[i - 1]
            delta = delta_previous * DERIVATIVES[previous_name](pre_activations[i - 1])

    return grads_W, grads_b


def predict_probabilities(X, weights, biases, activation_names):
    activations, _ = forward_pass(X, weights, biases, activation_names)
    return activations[-1]


def predict_classes(X, weights, biases, activation_names):
    probabilities = predict_probabilities(X, weights, biases, activation_names)
    return np.argmax(probabilities, axis=1)


def confusion_matrix(y_true, y_pred, n_classes=3):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[true_label, predicted_label] += 1
    return matrix


def macro_f1(y_true, y_pred, n_classes=3):
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    f1_scores = []
    for c in range(n_classes):
        true_positive = matrix[c, c]
        false_positive = matrix[:, c].sum() - true_positive
        false_negative = matrix[c, :].sum() - true_positive
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0.0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0.0
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)
    return float(np.mean(f1_scores))


class ANN:
    def __init__(self, hidden_sizes=(64, 32), hidden_activations=("relu", "tanh"),
                 learning_rate=0.01, momentum=0.9, l2_lambda=0.0001,
                 batch_size=64, max_epochs=100, patience=15, seed=SEED):
        if len(hidden_sizes) != len(hidden_activations):
            raise ValueError("hidden_sizes dan hidden_activations harus sama panjang")
        self.hidden_sizes = hidden_sizes
        self.hidden_activations = hidden_activations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        use_validation = X_val is not None and y_val is not None
        if use_validation:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val).astype(int)

        self.n_classes_ = int(y.max()) + 1
        self.layer_sizes_ = [X.shape[1]] + list(self.hidden_sizes) + [self.n_classes_]
        self.activation_names_ = list(self.hidden_activations) + ["softmax"]

        rng = np.random.default_rng(self.seed)
        weights, biases = initialize_parameters(self.layer_sizes_, self.activation_names_, self.seed)
        velocity_W = [np.zeros_like(W) for W in weights]
        velocity_b = [np.zeros_like(b) for b in biases]

        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_f1 = -1.0
        best_weights = None
        best_biases = None
        epochs_without_improvement = 0
        n_samples = X.shape[0]

        for epoch in range(self.max_epochs):
            order = rng.permutation(n_samples)
            X_shuffled = X[order]
            y_shuffled = y[order]

            for start in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[start:start + self.batch_size]
                y_batch = y_shuffled[start:start + self.batch_size]

                activations, pre_activations = forward_pass(X_batch, weights, biases, self.activation_names_)
                grads_W, grads_b = backward_pass(y_batch, activations, pre_activations,
                                                 weights, self.activation_names_, self.l2_lambda)

                for i in range(len(weights)):
                    velocity_W[i] = self.momentum * velocity_W[i] - self.learning_rate * grads_W[i]
                    velocity_b[i] = self.momentum * velocity_b[i] - self.learning_rate * grads_b[i]
                    weights[i] = weights[i] + velocity_W[i]
                    biases[i] = biases[i] + velocity_b[i]

            train_probabilities = predict_probabilities(X, weights, biases, self.activation_names_)
            history["train_loss"].append(cross_entropy_loss(y, train_probabilities, weights, self.l2_lambda))

            if use_validation:
                val_probabilities = predict_probabilities(X_val, weights, biases, self.activation_names_)
                history["val_loss"].append(cross_entropy_loss(y_val, val_probabilities, weights, self.l2_lambda))
                val_f1 = macro_f1(y_val, np.argmax(val_probabilities, axis=1), self.n_classes_)
                history["val_f1"].append(val_f1)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_weights = [W.copy() for W in weights]
                    best_biases = [b.copy() for b in biases]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        break

        if use_validation:
            self.weights_ = best_weights
            self.biases_ = best_biases
        else:
            self.weights_ = weights
            self.biases_ = biases
        self.history_ = history
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return predict_probabilities(X, self.weights_, self.biases_, self.activation_names_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path):
        arrays = {
            "layer_sizes": np.array(self.layer_sizes_),
            "activation_names": np.array(self.activation_names_),
        }
        for i in range(len(self.weights_)):
            arrays[f"W{i}"] = self.weights_[i]
            arrays[f"b{i}"] = self.biases_[i]
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls()
        model.layer_sizes_ = [int(n) for n in data["layer_sizes"]]
        model.activation_names_ = [str(n) for n in data["activation_names"]]
        n_layers = len(model.layer_sizes_) - 1
        model.weights_ = [data[f"W{i}"] for i in range(n_layers)]
        model.biases_ = [data[f"b{i}"] for i in range(n_layers)]
        model.n_classes_ = model.layer_sizes_[-1]
        return model
