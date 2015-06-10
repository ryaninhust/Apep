import numpy as np
import gnumpy as gnp

from act_functions import Softmax, ReLU
from datetime import datetime

class DNN(object):
    learning_rate = None
    batch_size = 0
    epochs = 0
    verbose = 0
    hidden_function = ReLU
    output_function = Softmax

    def __init__(self, layer_sizes, scale=0.05, verbose=1, l2=0.0001,
                 momentum=0.9, epochs=20, batch_size=256,dropouts=0.0,
                 learning_rate=0.01, learning_rate_decays=0.9):

        self.layer_sizes = layer_sizes
        self.scale = scale
        self.verbose = 1
        self.l2 = l2
        self.momentum = momentum

        self.epochs = epochs
        self.batch_size = batch_size
        self.dropouts = [dropouts for l in range(len(layer_sizes)-1)]

        self.learning_rate = learning_rate
        self.learning_rate_decays = learning_rate_decays

        shapes = [(layer_sizes[i-1], layer_sizes[i])
                  for i in range(1, len(layer_sizes))]

        self.biases = init_biases_matrix(layer_sizes)
        self.weights = init_weights_matrix(shapes, scale)
        self.rms_limits = [None for i in range(len(self.weights))]

        self.hidden_functions = [self.hidden_function for i in range(len(self.weights) - 1)]

        self.weight_grads_l2_norm = [gnp.ones(weight.shape) for weight in self.weights]
        self.bias_gradis_l2_norm = [gnp.ones(bias.shape) for bias in self.biases]
        self.weight_grads = [gnp.zeros(weight.shape) for weight in self.weights]
        self.bias_grads = [gnp.zeros(bias.shape) for bias in self.biases]


    def _onehot_encode(self, y):
        if len(y.shape) == 1:
            num_classes = y.max() + 1
            y_new = np.zeros((y.shape[0], num_classes), dtype=np.int)
            for index, label in enumerate(y):
                y_new[index][label] = 1
                y = y_new
        return y


    def _minibatches(self, X, y=None):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        while True:
            idx = np.random.randint(X.shape[0], size=(self.batch_size,))
            X_batch = X[idx]
            if y is not None:
                yield (X_batch, y[idx])
            else:
                yield X_batch


    def _adjust_learning_rate(self):
        self.learning_rate *= self.learning_rate_decays


    def _counter_num_mistakes(self, targets, predictions):
        if hasattr(targets, 'as_numpy_array'):
            targets = targets.as_numpy_array()
        if hasattr(predictions, 'as_numpy_array'):
            predictions = predictions.as_numpy_array()
        return np.sum(predictions.argmax(1) != targets.argmax(1))


    def save_weights(self, path):
        weights_path = "{0}_w".format(path)
        biases_path = "{0}_b".format(path)
        np.save(weights_path, self.weights)
        np.save(biases_path, self.biases)


    def load_weights(self, path):
        weights_path = "{0}_w".format(path)
        biases_path = "{0_b}".format(path)
        self.weights = np.load(weights_path)
        self.biases = np.load(biases_path)


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y = self._onehot_encode(y)
        iters_per_epoch = X.shape[0] / self.batch_size
        for epoch, err in enumerate(
            self.fine_tune(self._minibatches(X, y), iters_per_epoch, self.epochs)
        ):
            print"epoch:{0}    ein:{1}".format(epoch, err)


    def predict_proba(self, X):
        prob_result = []
        for i, _input in enumerate(np.array_split(X, 100)):
            if _input is not isinstance(_input, gnp.garray):
                _input = gnp.garray(_input)
            self.dropouts = [0.0 for l in self.dropouts]
            output = self.feed_forward(_input)
            prob_result.append(output.as_numpy_array())
        prob_result = np.concatenate(prob_result)
        return prob_result


    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


    def score(self, X, y):
        predictions = self.predict(X)
        error_count = 0
        for i, y_p in enumerate(y):
            if y_p != predictions[i]:
                error_count += 1
        return error_count / float(X.shape[0])


    def fine_tune(self, mini_batch, iters_per_epoch, epochs):
        s_time = datetime.now()
        for epoch in range(epochs):
            total_errors = 0
            total_inputs = 0
            for i in range(iters_per_epoch):
                X, y = mini_batch.next()
                err, y_pred = self.epoch_update(X, y)
                total_inputs += X.shape[0]
                total_errors += self._counter_num_mistakes(y, y_pred)
            print datetime.now() - s_time
            yield total_errors / float(total_inputs)


    def scale_gradient(self, scale):
        for i in range(len(self.weights)):
            self.weight_grads[i] *= scale
            self.bias_grads[i] *= scale


    def feed_forward(self, input_batch):
        if not isinstance(input_batch, gnp.garray):
            input_batch = gnp.garray(input_batch)
        weights_to_stop = len(self.weights)
        self.state = [input_batch * (gnp.rand(*input_batch.shape) > self.dropouts[0])]

        for i in range(min(len(self.weights) -1, weights_to_stop)):
            do_factor = 1.0 / (1.0-self.dropouts[i])
            linear_outputs = gnp.dot(self.state[-1]*do_factor, self.weights[i]) + self.biases[i]
            act_outputs = self.hidden_functions[i].activate(linear_outputs)
            self.state.append(act_outputs*(gnp.rand(*act_outputs.shape) > self.dropouts[i+1]))

        if weights_to_stop >= len(self.weights):
            do_factor = 1.0 / (1.0-self.dropouts[-1])
            self.state.append(gnp.dot(self.state[-1]*do_factor, self.weights[-1]) + self.biases[-1])
            self.acts = self.output_function.activate(self.state[-1])
            return self.acts

        return self.state[weights_to_stop]


    def back_propagation(self, output_residual):
        feed_forward_state = self.state
        error_residual = [None for i in range(len(self.weights))]
        error_residual[-1] = output_residual
        for i in reversed(range(len(self.weights) -1)):
            residual = gnp.dot(error_residual[i+1], self.weights[i+1].T) * self.hidden_functions[i].residual(feed_forward_state[i+1])
            error_residual[i] = residual
        return error_residual


    def gradients(self, feed_forward_state, error_residual):
        for i in range(len(self.weights)):
            a = (gnp.dot(feed_forward_state[i].T, error_residual[i]),
                   error_residual[i].sum(axis=0))
            yield a


    def feed_back(self, input_batch, target_batch):
        output_result = self.feed_forward(input_batch)
        output_residual = -self.output_function.residual(target_batch, self.state[-1], output_result)
        error = self.output_function.error(target_batch, self.state[-1], output_result)
        error_residual = self.back_propagation(output_residual)
        return error_residual, output_result, error


    def epoch_update(self, input_batch, output_batch):
        this_batch_size = input_batch.shape[0]

        if not isinstance(input_batch, gnp.garray):
            input_batch = gnp.garray(input_batch)
        if not isinstance(output_batch, gnp.garray):
            output_batch = gnp.garray(output_batch)

        error_residual, output_result, error = self.feed_back(input_batch, output_batch)


        for i, (w_grad, b_grad) in enumerate(self.gradients(self.state, error_residual)):
            self.weight_grads_l2_norm[i] += (w_grad/this_batch_size - self.l2*self.weights[i]) ** 2
            self.bias_gradis_l2_norm[i] += (b_grad/this_batch_size) ** 2
            w_factor = 1 / gnp.sqrt(self.weight_grads_l2_norm[i])
            b_factor = 1 / gnp.sqrt(self.bias_gradis_l2_norm[i])
            self.weight_grads[i] = self.learning_rate * w_factor * (
                w_grad/this_batch_size - self.l2*self.weights[i])
            self.bias_grads[i] = (self.learning_rate*b_factor/this_batch_size) * b_grad

        for i in range(len(self.weights)):
            self.weights[i] += self.weight_grads[i]
            self.biases[i] += self.bias_grads[i]

        return error, output_result


    def update_weights(self, new_weights, new_biases, origin_weights, origin_biases,
                       weight_grads, bias_grads):
        for i in range(len(new_weights)):
            new_weights[i] = origin_weights[i] + weight_grads[i]
            new_biases[i] = origin_biases[i] + bias_grads[i]


    def constrain_weights(self):
        for i, rms_limit in enumerate(self.rms_limits):
            if not rms_limit:
                continue
            W = self.weights[i]
            rms_scale = rms_limit / gnp.sqrt(gnp.mean(W*W, axis=0))
            limit_rms = W * (1+(rms_scale < 1) * (rms_scale - 1))
            self.weights[i] = limit_rms


def init_weights_matrix(shapes, scale):
    weights = []
    for shape in shapes:
        fan_in = shape[0]
        W = scale * np.random.randn(*shape)
        for i in range(shape[1]):
            perm = np.random.permutation(shape[0])
            W[perm[fan_in:], i] *= 0
        weights.append(gnp.garray(W))
    return weights


def init_biases_matrix(layer_sizes):
    return [gnp.garray(0*np.random.rand(1, layer_sizes[i]))
            for i in range(1, len(layer_sizes))]


