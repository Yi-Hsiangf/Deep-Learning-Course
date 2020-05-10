import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=100):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval
        self.learning_rate = 0.01
        self.size = hidden_size

        # Model parameters initialization
        # Please initiate your network parameters here.
        
        # Init weight
        
        self.w1 = np.array([[random.random() for i in range(2)] for j in range(hidden_size)])
        self.w2 = np.array([[random.random() for i in range(hidden_size)] for j in range(hidden_size)])
        self.w3 = np.array([random.random() for i in range(hidden_size)])
        
        # Store the output of neurons for back propagation
        self.Z1 = [0.] * hidden_size
        self.Z2 = [0.] * hidden_size
        
         # Store the gradient of weights
        self.w1_gradient = np.array([[0.] * 2] * hidden_size)
        self.w2_gradient = np.array([[0.] * hidden_size] * hidden_size)
        self.w3_gradient = np.array([0.] * hidden_size)
       
        ...

    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

    def forward(self, inputs):
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results.
        """
        output = []
        for input in inputs:
            #print(input)
            for id, w in enumerate(self.w1):
                self.Z1[id] = sigmoid(np.dot(w, input))
    
            for id, w in enumerate(self.w2):
                self.Z2[id] = sigmoid(np.dot(w, self.Z1))
            
            self.y = sigmoid(np.dot(self.w3, self.Z2))

            output.append(self.y)
        
        return output

    def backward(self):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        """
        
        # BACK PROPAGATION
        # Compute the gradient for W3
        for id, w in enumerate(self.w3_gradient):
            self.w3_gradient[id] = (self.y - self.label) * der_sigmoid(self.y) * self.Z2[id]
        
        # Compute the gradient for W2
        for idi, wi in enumerate(self.w2_gradient):
            for idj, wj in enumerate(self.w2_gradient[idi]):
                self.w2_gradient[idi][idj] = (self.y - self.label) * der_sigmoid(self.y) * self.w3[idi] * der_sigmoid(self.Z2[idi]) * self.Z1[idj]
        
        # Compute the gradient for W1
        for idi, wi in enumerate(self.w1_gradient):
            for idj, wj in enumerate(self.w1_gradient[idi]):
                self.w1_gradient[idi][idj] = 0
                for idk, wk in enumerate(self.Z2):
                    self.w1_gradient[idi][idj] += (self.y - self.label) * der_sigmoid(self.y) * self.w3[idk] * der_sigmoid(self.Z2[idk]) * self.w2[idk][idi]\
                                                  * der_sigmoid(self.Z1[idi]) * self.input_data[idj]

        # GRADIENT DESCENT
        # For W3
        for id, w in enumerate(self.w3):
            self.w3[id] = self.w3[id] - self.learning_rate * self.w3_gradient[id]

        # For W2
        for idi, wi in enumerate(self.w2):
            for idj, wj in enumerate(self.w2[idi]):
                self.w2[idi][idj] = self.w2[idi][idj] - self.learning_rate * self.w2_gradient[idi][idj]
        
        # For W1
        for idi, wi in enumerate(self.w1):
            for idj, wj in enumerate(self.w1[idi]):
                self.w1[idi][idj] = self.w1[idi][idj] - self.learning_rate * self.w1_gradient[idi][idj]
    

    def backward2(self):
        # Store the gradient of weights
        self.w1_delta = np.array([[0.] * 2] * self.size)
        self.w2_delta = np.array([[0.] * self.size] * self.size)
        self.w3_delta = np.array([0.] * self.size)

        for id, w in enumerate(self.w3_gradient):
            self.w3_delta[id] = -1 * np.reshape(self.error, (1,)) * der_sigmoid(self.y)
            self.w3_gradient[id] =  self.w3_delta[id] * self.Z2[id]

        for idi, wi in enumerate(self.w2_gradient):
            for idj, wj in enumerate(self.w2_gradient[idi]):
                self.w2_delta[idi][idj] = self.w3[idi] * self.w3_delta[idi]  * der_sigmoid(self.Z2[idi])
                self.w2_gradient[idi][idj] =  self.w2_delta[idi][idj] * self.Z1[idj]
                # print("w2 gradient", self.w2_gradient[idi][idj])
       
        for idi, wi in enumerate(self.w1_gradient):
            for idj, wj in enumerate(self.w1_gradient[idi]):
                for idk, wk in enumerate(self.Z2):
                    self.w1_delta[idi][idj] += self.w2[idk][idi] * self.w2_delta[idk][idi] * der_sigmoid(self.Z1[idi])
                self.w1_gradient[idi][idj] = self.w1_delta[idi][idj] * self.input_data[idj]
        
        # GRADIENT DESCENT
        # For W3
        for id, w in enumerate(self.w3):
            self.w3[id] = self.w3[id] + self.learning_rate * self.w3_gradient[id]
       
        # For W2
        for idi in range(len(self.w2)):
            for idj in range(len(self.w2[idi])):
                # print(idi, idj)
                # print("w2 value before", self.w2)
                self.w2[idi][idj] = self.w2[idi][idj] + self.learning_rate * self.w2_gradient[idi][idj]
                # print("w2 value", self.w2)
        
        # For W1
        for idi, wi in enumerate(self.w1):
            for idj, wj in enumerate(self.w1[idi]):
                self.w1[idi][idj] = self.w1[idi][idj] + self.learning_rate * self.w1_gradient[idi][idj]
         
    def print_weight(self):
        print("w1:")
        print(self.w1)
        print(self.w1_delta)
        print()

        print("w2:")
        print(self.w2)
        print(self.w2_delta)
        print()

        print("w3:")
        print(self.w3)
        print(self.w3_delta)
        print()
    
    
    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]
        self.input_num = n

        for epochs in range(self.num_step):
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
                self.input_data = np.reshape(inputs[idx:idx+1, :], (2, ))
                self.output = self.forward(inputs[idx:idx+1, :])

                self.error = self.output - labels[idx:idx+1, :]
                self.label = np.reshape(labels[idx:idx+1, :], (1, ))
                self.backward2()
                """ if epochs == 9:
                   self.print_weight() """

            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)

        print('Training finished')
        self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]

        error = 0.0
        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            error += abs(result - labels[idx:idx+1, :])
            
         
        print("error: ", error)
        error /= n
        print('accuracy: %.2f' % ((1 - error)*100) + '%')
        print('')


if __name__ == '__main__':
    data, label = GenData.fetch_data('Linear', 5)
    
    """  data = np.array([[0.,0.], [0., 1.]])
    label = np.array([[0], [1]]) """
    
    """ input_size = data.shape[0]
    print(input_size)
    print(data) """

    net = SimpleNet(5, num_step=100000)
    net.train(data, label)
   
    output = net.forward(data)
    
    pred_result = []

    pred_result = np.round(output)
    print(output)
    print(pred_result)
    pred_result = np.array(pred_result)
    SimpleNet.plot_result(data, label, pred_result.T)
