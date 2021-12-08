from functools import reduce
import random
import numpy as np
import torch
from typing import Any, Tuple, Callable, List, NamedTuple
from torch.nn import functional
from torch.autograd.functional import jacobian
from torch.nn.modules.activation import LogSoftmax, Softmax
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import NLLLoss
from torch.nn.modules.pooling import AdaptiveMaxPool2d, MaxPool2d
import torchvision
import tqdm


# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    paddings: Tuple[int, ...] = (0, 0, 0)
    dense_hiddens: Tuple[int, ...] = (256, 256)

def pretty_print(title: str, data : Any):
    print(f'\n\n{title} :\n')
    print(data)
    print('\n\n')

def pretty_print_list(title: str, list: List):
    print(f'\n\n{title} :\n')
    for el in list:
        print(el)
    print('\n\n')

# Pytorch preliminaries
def gradient_norm(function: Callable, *tensor_list: List[torch.Tensor]) -> float:
    output = function(*tensor_list)
    # Trigger gradient computation
    output.backward(retain_graph=True)

    # Retrieve gradient for each tensor input
    grad = list(map(lambda ts : ts.grad.flatten().tolist(), tensor_list))
    # Flatten the gradients into 1 long list
    grad_flat = list(reduce(lambda cumul, ts : cumul + ts, grad, []))

    # Euclidian norm of flat gradient
    norm = np.linalg.norm(grad_flat)
    return norm

def jacobian_norm(function: Callable, input_tensor: torch.Tensor) -> float:
    # Compute jacobian matrix and flatten it
    jacobian_flat = jacobian(function, input_tensor).flatten().tolist()

    # Frobenius norm on a matrix is just the Euclidian norm on the flattened
    # matrix
    norm = np.linalg.norm(jacobian_flat)
    return norm

class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 datapath: str = './data',
                 n_classes: int = 10,
                 lr: float = 0.0001,
                 batch_size: int = 128,
                 activation_name: str = "relu",
                 normalization: bool = True):
        pretty_print('Type', network_type)
        pretty_print('Net Config', net_config)
        pretty_print('Classes', n_classes)
        self.train, self.valid, self.test = self.load_dataset(datapath)
        if normalization:
            self.train, self.valid, self.test = self.normalize(self.train, self.valid, self.test)
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        input_dim = self.train[0][0].shape
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], net_config,
                                           n_classes, activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], net_config, n_classes, activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = 1e-9

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': [],
                           'train_gradient_norm': []}

    @staticmethod
    def load_dataset(datapath: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        trainset = torchvision.datasets.FashionMNIST(root=datapath,
                                                     download=True, train=True)
        testset = torchvision.datasets.FashionMNIST(root=datapath,
                                                    download=True, train=False)

        X_train = trainset.data.view(-1, 1, 28, 28).float()
        y_train = trainset.targets

        X_ = testset.data.view(-1, 1, 28, 28).float()
        y_ = testset.targets

        X_val = X_[:2000]
        y_val = y_[:2000]

        X_test = X_[2000:]
        y_test = y_[2000:]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration, n_classes: int,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param n_classes: The number of classes to predict.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        hidden_layers = []
        last_layer_out_features = input_dim
        for dim in net_config.dense_hiddens:
            hidden_layers += [
                Linear(last_layer_out_features, dim),
                activation
            ]

            last_layer_out_features = dim
        
        # At this point, `last_layer_out_features` contains the number of
        # features in the last hidden layer. This is used as the number of
        # input features in the last layer.
        return Sequential(
            Flatten(),
            *hidden_layers,
            Linear(last_layer_out_features, n_classes),
            Softmax(dim = 1)
        )

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration, n_classes: int,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param n_classes: The number of classes to predict.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        # Format netconfig so it's a list of tuples, one per hidden layer
        net_config_zip = list(enumerate(zip(*net_config[:-1])))

        conv_layers = []
        last_n_channel = in_channels
        for (i, config) in net_config_zip:
            # unpacking the configuration for a single convolutional layer
            n_channel, k_size, stride, pad, _ = config
            conv_layers += [
                Conv2d(last_n_channel, n_channel, k_size, stride, pad),
                activation
            ]

            if i < len(net_config_zip) - 1:
                conv_layers.append(MaxPool2d(kernel_size=2))
            else:
                # pooling on last convolutional layer is different
                conv_layers.append(AdaptiveMaxPool2d((4, 4)))

            last_n_channel = n_channel
            
        conn_layers = [Flatten()]
        # this is because the last pooling has size (4, 4)
        last_conn_out_features = last_n_channel * 4 * 4
        for dim in net_config.dense_hiddens:
            conn_layers += [
                Linear(last_conn_out_features, dim),
                activation
            ]

            last_conn_out_features = dim

        # At this point, `last_conn_out_features` contains the number of
        # features in the last fully-connected layer. This is used as the
        # number of input features in the last layer.
        return Sequential(
            *conv_layers,
            *conn_layers,
            Linear(last_conn_out_features, n_classes),
            Softmax(dim = 1)
        )

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        return {
            'relu' : torch.nn.ReLU(),
            'tanh' : torch.nn.Tanh(),
            'sigmoid' : torch.nn.Sigmoid()
        }[activation_str]

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return functional.one_hot(y, self.n_classes)

    def compute_loss_and_accuracy(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Predictions given by network with current weights
        pretty_print('X', X.stride())
        pretty_print('y', y.stride())
        pretty_print_list('y list', y)
        pretty_print_list('Module', self.network.named_modules(remove_duplicate=False))
        predictions = self.network(X).clip(self.epsilon, 1 - self.epsilon)
        pretty_print_list('Predictions', predictions)

        loss = cross_entropy(predictions, y.float())
        pretty_print('Loss', loss)
        # Trigger gradient computation
        loss.backward(retain_graph=True)

        _, prediction_choices = torch.max(predictions, dim = 1)
        pretty_print_list('Prediction Choices', prediction_choices)
        _, y_choices = torch.max(y, dim = 1)
        pretty_print_list('Y Choices', y_choices)

        total = correct = 0
        for actual, expected in zip(prediction_choices, y_choices):
            total += 1

            if actual == expected:
                correct += 1
        
        accuracy = correct / total
        pretty_print('Accuracy', accuracy)

        return (loss, accuracy)
        

    @staticmethod
    def compute_gradient_norm(network: torch.nn.Module) -> float:
        gradients = [p.grad.flatten().tolist() for p in network.parameters()]
        # Flatten the gradients into 1 long list
        grad_flat = list(reduce(lambda cumul, ts : cumul + ts, gradients, []))

        # Euclidian norm of flat gradient
        norm = np.linalg.norm(grad_flat)
        return norm

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        # TODO WRITE CODE HERE
        pass

    def log_metrics(self, X_train: torch.Tensor, y_train_oh: torch.Tensor,
                    X_valid: torch.Tensor, y_valid_oh: torch.Tensor) -> None:
        self.network.eval()
        with torch.inference_mode():
            train_loss, train_accuracy = self.compute_loss_and_accuracy(X_train, y_train_oh)
            valid_loss, valid_accuracy = self.compute_loss_and_accuracy(X_valid, y_valid_oh)
        self.train_logs['train_accuracy'].append(train_accuracy)
        self.train_logs['validation_accuracy'].append(valid_accuracy)
        self.train_logs['train_loss'].append(float(train_loss))
        self.train_logs['validation_loss'].append(float(valid_loss))

    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train, y_train = self.train
        y_train_oh = self.one_hot(y_train)
        X_valid, y_valid = self.valid
        y_valid_oh = self.one_hot(y_valid)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        self.log_metrics(X_train[:2000], y_train_oh[:2000], X_valid, y_valid_oh)
        for epoch in tqdm.tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_train_oh[self.batch_size * batch:self.batch_size * (batch + 1), :]
                gradient_norm = self.training_step(minibatchX, minibatchY)
            # Just log the last gradient norm
            self.train_logs['train_gradient_norm'].append(gradient_norm)
            self.log_metrics(X_train[:2000], y_train_oh[:2000], X_valid, y_valid_oh)
        return self.train_logs

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # TODO WRITE CODE HERE
        pass

    @staticmethod
    def normalize(train: Tuple[torch.Tensor, torch.Tensor],
                  valid: Tuple[torch.Tensor, torch.Tensor],
                  test: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                                    Tuple[torch.Tensor, torch.Tensor],
                                                                    Tuple[torch.Tensor, torch.Tensor]]:
        # Computing mean and standard deviation on train data ONLY
        mean = torch.mean(train[0], dim=0)
        std = torch.std(train[0], dim = 0)

        pretty_print('Train', train[0][0].stride())
        pretty_print('Mean', mean.stride())
        pretty_print('Std', std.stride())

        return (
            (train[0].sub(mean).div(std), train[1]),
            (valid[0].sub(mean).div(std), valid[1]),
            (test[0].sub(mean).div(std), test[1])
        )

    def test_equivariance(self):
        from functools import partial
        test_im = self.train[0][0]/255.
        conv = torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=0)
        fullconv_model = lambda x: torch.relu(conv((torch.relu(conv((x))))))
        model = fullconv_model

        shift_amount = 5
        shift = partial(torchvision.transforms.functional.affine, angle=0,
                        translate=(shift_amount, shift_amount), scale=1, shear=0)
        rotation = partial(torchvision.transforms.functional.affine, angle=90,
                           translate=(0, 0), scale=1, shear=0)

        # TODO CODE HERE
        pass
