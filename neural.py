import numpy
import random

DEFAULT_THRESHOLD = -0.1
THRESHOLD_INPUT = 1.0

ERROR_SATISFACTION = 0.001

NI = 0.01 #learning rate

SIGMOIDAL_SLOPE = 1.0

WEIGHT_RANDOM_LIMIT = 0.1

#module initializations
random.seed()

class NeuralLayer:
    def __init__(self, my_size, previous_layer_size):
        self.size = my_size
        self.weights_per_neuron = previous_layer_size + 1 #1 for threshold
        self.weights = []
        for i in xrange(my_size):
            neuron_weights = [random.random() * WEIGHT_RANDOM_LIMIT \
                              for j in xrange(previous_layer_size)]
            neuron_weights.append(random.random()*DEFAULT_THRESHOLD)
            self.weights.append(neuron_weights)
        self.weights = numpy.array(self.weights, numpy.float)

        self.inputs = numpy.array([0.] * self.weights_per_neuron, numpy.float)
        self.outputs = numpy.array([0.] * my_size, numpy.float)
        self.errors = numpy.array([0.] * my_size, numpy.float)
        
    def activation_func(self, x):
        #Sigmoidal function
        #result = 1 / (1 + numpy.exp(-(SIGMOIDAL_SLOPE * x)))
        result = numpy.tanh(x)
        return result

    def activate(self, input_vector):
        input_vector = list(input_vector)
        self.inputs = numpy.array(input_vector, numpy.float)
        inputs = numpy.concatenate((self.inputs, [THRESHOLD_INPUT]))
        for i in xrange(self.size):
            assert(inputs.size == self.weights[i].size)
            weighted_inputs_sum = numpy.inner(self.weights[i], inputs)
            self.outputs[i] = self.activation_func(weighted_inputs_sum)
        return self.outputs

    def update_weights(self, errors):
        self.errors = numpy.array(errors, numpy.float)
        assert(self.errors.size == self.size)
        inputs = numpy.concatenate((self.inputs, [THRESHOLD_INPUT]))
        dw = NI * numpy.outer(self.errors, inputs)
        assert(dw.shape == self.weights.shape)
        #dw[:,-1:] = numpy.array(dw.shape[0] * [0.0]) #don't update bias
        self.weights = self.weights + dw

class InputLayer(NeuralLayer):
    def __init__(self, my_size):
        NeuralLayer.__init__(self, my_size, 0)
        
    def activate(self, input_vector):
        self.outputs = numpy.array(input_vector, numpy.float)
        
class NeuralNetwork:
    "Represents a feed-forward network"
    def __init__(self, *layer_sizes):
        assert(len(layer_sizes) >= 2) #there must be at least an input and an output layer
        self.num_of_layers = len(layer_sizes)
        self.layers = [InputLayer(layer_sizes[0])]
        for i in xrange(1, self.num_of_layers):
            self.layers.append(self.create_layer(layer_sizes[i], \
                                                 layer_sizes[i-1]))

        self.output_layer = self.layers[self.num_of_layers-1]

    def create_layer(self, size, previous_layer_size):
        return NeuralLayer(size, previous_layer_size)

    def activate(self, input_vector):
        assert(len(input_vector) == self.layers[0].size)
        self.layers[0].activate(input_vector)
        for i in xrange(1, self.num_of_layers):
            self.layers[i].activate(self.layers[i-1].outputs)

##    def train(self, input_vector, expected_output_vector):
##        assert(len(expected_output_vector) == self.output_layer.size)
##        self.activate(input_vector)
##        errors = numpy.array(expected_output_vector, numpy.float) - \
##                 self.output_layer.outputs
##        self.output_layer.update_weights(errors)

class BP_NeuralNetwork(NeuralNetwork):
    "Back-Propagation neural network"
    def __init__(self, *layer_sizes):
        NeuralNetwork.__init__(self, *layer_sizes)

    def train(self, input_vector, expected_output_vector):
        #print "Training:\ninput = ", input_vector, "\nexpected output = ", expected_output_vector
        assert(len(expected_output_vector) == self.output_layer.size)
        self.activate(input_vector)

        #calculate errors for output layer
        t = numpy.array(expected_output_vector, numpy.float)
        out = self.output_layer.outputs
        #errors = (t - out) * out * (1 - out)
        errors = (t - out) * (1.0 - out*out)
        self.output_layer.update_weights(errors)
        
        for i in xrange(self.num_of_layers-2, 0, -1):
            out = self.layers[i].outputs
            next_layer_errors = self.layers[i+1].errors
            next_layer_weights = self.layers[i+1].weights[:,:-1]
            errors = numpy.inner(next_layer_weights.transpose(), \
                                          next_layer_errors)
            #errors = out * (1 - out) * errors
            errors = (1.0 - out*out) * errors
            #print errors
            self.layers[i].update_weights(errors)

        total_error = numpy.sum(0.5 * (t - self.output_layer.outputs)**2)
        print "Error = ", total_error
        return total_error

def main():
    #nnw = NeuralNetwork(2,1)
    nnw = BP_NeuralNetwork(2, 4, 4, 1)

    #train AND function
    for i in xrange(100000):
        error = 0.0
        error += nnw.train([0,1], [0])
        error += nnw.train([0,0], [0])
        error += nnw.train([1,1], [1])
        error += nnw.train([1,0], [0])
        if (error < ERROR_SATISFACTION):
            break
    
    nnw.activate([1,0])
    print nnw.output_layer.outputs #expected 0
    nnw.activate([1,1])
    print nnw.output_layer.outputs #expected 1
    nnw.activate([1.2,0.9])
    print nnw.output_layer.outputs #expected 1
    nnw.activate([0.1,1])
    print nnw.output_layer.outputs #expected 0

    nnw = BP_NeuralNetwork(8, 4, 2)
    for i in xrange(20000):
        error = 0.0
        error += nnw.train([0, 1, 1, 0, 0, 0, 0, 0], [1, 0])
        error += nnw.train([0, 1, 1, 0, 0, 0, 1, 1], [0, 1])
        if (error < ERROR_SATISFACTION):
            break
    
    nnw.activate([0, 1, 1, 0, 0, 0, 0, 0])
    print nnw.output_layer.outputs
    nnw.activate([0, 1, 1, 0, 0, 0, 1, 1])
    print nnw.output_layer.outputs

if __name__ == '__main__':
    main()
