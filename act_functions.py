import numpy as np
import gnumpy as gnp

class Sigmoid(object):
    @classmethod
    def activate(cls, net_input):
        return net_input.sigmoid()
    @classmethod
    def error(cls, targets, net_input, out_put = None):
        return (net_input.log_1_plus_exp()-targets*net_input).sum()

    @classmethod
    def residual(cls, net_input):
        return net_input * (1-net_input)


class ReLU(object):
    @classmethod
    def activate(cls, net_input):
        return net_input*(net_input > 0)
    @classmethod
    def residual(cls, net_output):
        return net_output > 0


class Maxout(object):

    def __init__(self):
        self.maxout_index_matrix = None

    def activate(self, net_input, group_size):
        net_input = net_input.as_numpy_array()
        self.maxout_index_matrix = np.zeros((net_input.shape[0], net_input.shape[1]/group_size))
        result_matrix = np.zeros((net_input.shape[0], net_input.shape[1]/group_size))
        net_input = net_input.tolist()
        for j in range(result_matrix.shape[0]):
            for ii, i in enumerate(range(0, result_matrix.shape[1]*3, group_size)):
                segment = net_input[j][i:i+group_size]
                value = max(segment)
                result_matrix[j][ii] = value
                self.maxout_index_matrix[j][ii] = segment.index(value) + i
        return gnp.garray(result_matrix)

    def residual(self, net_input):
        return 1
    def expand_residual(self, residual):
        residual = residual.as_numpy_array()
        residual_matrix = np.zeros((residual.shape[0], residual.shape[1]*3))
        for index, i in enumerate(self.maxout_index_matrix):
            for index_j, j in enumerate(i):
                residual_matrix[index][j] = residual[index][index_j]
        return gnp.garray(residual_matrix)


class Softmax(object):

    @classmethod
    def activate(cls, net_input):
        Zshape = (net_input.shape[0],1)
        output = net_input - net_input.max(axis=1).reshape(*Zshape)
        output = output.exp()
        return output/output.sum(axis=1).reshape(*Zshape)

    @classmethod
    def residual(cls, targets, net_input, output = None):
        if output == None:
            output = cls.activate(net_input)
        return output - targets

    @classmethod
    def error(cls, targets, net_input, output = None):
        ntInpt = net_input - net_input.max(axis=1).reshape(net_input.shape[0],1)
        logZs = ntInpt.exp().sum(axis=1).log().reshape(-1,1)
        err = targets*(ntInpt - logZs)
        return -err.sum()


class Tanh(object):
    def activate(self, net_input):
        return gnp.tanh(net_input)
    def residual(self, net_input):
        return 1-net_input*net_input
