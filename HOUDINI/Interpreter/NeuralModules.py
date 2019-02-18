import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from abc import abstractmethod
import json
import os


class SaveableNNModule(nn.Module):
    def __init__(self, params_dict: dict = None):
        self.params_dict = params_dict
        super(SaveableNNModule, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

    def save(self, directory):
        if directory[-1] == '/':
            directory = directory[:-1]

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = "{}/{}.pth".format(directory, self.name)
        torch.save(self.state_dict(), file_path)

        if self.params_dict is not None:
            if "output_activation" not in self.params_dict or self.params_dict["output_activation"] is None:
                self.params_dict["output_activation"] = "None"
            elif self.params_dict["output_activation"] == F.sigmoid:
                self.params_dict["output_activation"] = "sigmoid"
            elif type(self.params_dict["output_activation"]) == nn.Softmax:
                self.params_dict["output_activation"] = "softmax"
            else:
                raise NotImplementedError

            jsond = json.dumps(self.params_dict)
            f = open("{}/{}.json".format(directory, self.name), "w")
            f.write(jsond)
            f.close()

    @staticmethod
    def create_and_load(directory, name, new_name=None):
        if new_name is None:
            new_name = name

        with open('{}/{}.json'.format(directory, name)) as json_data:
            params_dict = json.load(json_data)
            params_dict["name"] = new_name

            if params_dict["output_activation"] == "None":
                params_dict["output_activation"] = None
            elif params_dict["output_activation"] == "sigmoid":
                params_dict["output_activation"] = F.sigmoid
            elif params_dict["output_activation"] == "softmax":
                params_dict["output_activation"] = nn.Softmax(dim=1)
            else:
                raise NotImplementedError

        new_fn, _ = get_nn_from_params_dict(params_dict)
        # print(new_name)
        # new_fn = new_fn_dict[new_name]
        new_fn.load("{}/{}.pth".format(directory, name))
        new_fn.eval()
        return new_fn


class NetCNN(SaveableNNModule):
    def __init__(self, name, input_dim, input_ch):
        """
        :param output_activation: [None, F.softmax, F.sigmoid]
        """
        super(NetCNN, self).__init__()

        self.name = name
        # self.layer_sizes = [64, 32]
        self.layer_sizes = [32, 64]

        self.conv1 = nn.Conv2d(input_ch, self.layer_sizes[0], kernel_size=5)
        conv1_output_dim = self.cnn_get_output_dim(input_dim, 5, stride=1, padding=0)
        pool1_output_dim = self.cnn_get_output_dim(conv1_output_dim, 2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(self.layer_sizes[0], self.layer_sizes[1], kernel_size=5)
        conv2_output_dim = self.cnn_get_output_dim(pool1_output_dim, kernel_size=5, stride=1, padding=0)
        self.pool2_output_dim = self.cnn_get_output_dim(conv2_output_dim, 2, stride=2, padding=0)

        self.conv2_drop = nn.Dropout2d()
        """
        self.fc1 = nn.Linear(self.pool2_output_dim ** 2 * self.layer_sizes[1], 1024)
        self.bn1 = nn.BatchNorm1d(self.layer_sizes[2])
        if self.output_dim is not None:
            self.fc2 = nn.Linear(1024, output_dim)
        """

    def cnn_get_output_dim(self, w1, kernel_size, stride, padding=0):
        w2 = (w1 - kernel_size + 2*padding) // stride + 1
        return w2

    def forward(self, x):
        if type(x) == tuple:
            x = x[1]

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return x


class NetMLP(SaveableNNModule):
    def __init__(self, name, input_dim, output_dim, output_activation=None, hidden_layer=True):
        """
        :param
        :param output_activation: [None, F.softmax, F.sigmoid]
        """
        super(NetMLP, self).__init__()
        self.output_dim = output_dim
        self.name = name
        self.output_activation=output_activation
        self.hidden_layer = hidden_layer
        # fc1_size = 300
        if self.hidden_layer:
            fc1_size = 1024
            self.fc1 = nn.Linear(input_dim, fc1_size)
            self.bn1 = nn.BatchNorm1d(fc1_size)

        if self.output_dim is not None:
            if self.hidden_layer:
                self.fc2 = nn.Linear(fc1_size, output_dim)
            else:
                self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x, x1=None):
        if type(x) == tuple:
                x = x[1]
        if list(x.shape).__len__() == 4:  # if it's 2d, flatten to 1d
            x = x.view(x.shape[0], -1)

        # If one of the inputs (x or x1) is a constant, we enlargen it to [batch_size, _], so that it can be concatenated
        if x1 is not None:
            if type(x1) == tuple:
                x1 = x1[1]

            dim0_x = x.shape[0]
            dim0_x1 = x1.shape[0]
            dim0 = max((dim0_x, dim0_x1))

            if dim0_x == 1 and dim0 > 1:
                dim1_x = x.shape[1]
                new = torch.ones((dim0, dim1_x))
                new = Variable(new).cuda() if torch.cuda.is_available() else Variable(new)
                x = new*x

            if dim0_x1 == 1 and dim0 > 1:
                dim1_x1 = x1.shape[1]
                new = torch.ones((dim0, dim1_x1))
                new = Variable(new).cuda() if torch.cuda.is_available() else Variable(new)
                x1 = new * x1

            # x1.shape[0] == 1

            x = torch.cat((x, x1), dim=1)

        # FC Layer 1
        if self.hidden_layer:
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            x = F.dropout(x, training=self.training)

        # FC Layer 2
        if self.output_dim is not None:
            x_logits = self.fc2(x)
            output = x_logits if self.output_activation is None else self.output_activation(x_logits)
            return x_logits, output
        else:
            return x


class NetRNN(SaveableNNModule):
    def __init__(self, name, input_dim, hidden_dim, output_dim=None, output_activation=None, output_sequence=False):
        """
        A function which goes from a list of items to a single hidden representation
        """
        super(NetRNN, self).__init__()
        self.name = name
        self.hidden_dim = hidden_dim

        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim)
        self.hidden = None # a placeholder, used for the hidden state of the lstm
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.output_sequence = output_sequence
        if self.output_dim is not None:  # then we need an mlp at the end
            self.mlp = NetMLP(name="{}_mlp".format(self.name), input_dim=hidden_dim, output_dim=self.output_dim,
                              output_activation=self.output_activation, hidden_layer=False)

    def reset_hidden_state(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        t1 = torch.zeros(1, batch_size, self.hidden_dim)
        t2 = torch.zeros(1, batch_size, self.hidden_dim)

        if torch.cuda.is_available():
            #print("converting to cuda")
            t1.cuda()
            t2.cuda()

        var1 = Variable(t1)
        var2 = Variable(t2)
        self.hidden = (var1.cuda(), var2.cuda()) if torch.cuda.is_available() else (var1, var2)

    def forward(self, x):
        # if necessarry, concatenate from list to tensor
        if type(x) == list:
            if x.__len__() > 0 and type(x[0]) == tuple:
                x = [i[1] for i in x]
            x = [torch.unsqueeze(a, dim=1) for a in x]
            x = torch.cat(x, dim=1)

        # find out the batch_size and use it to reset the hidden state
        batch_size = x.data.shape[0]
        self.reset_hidden_state(batch_size)

        # transpose to a suitable form [list_size, batsh_size, items]
        x = torch.transpose(x, 0, 1)

        # apply rnn list_size number of times
        # lstm_output: [seq_len, batch, hidden_size * num_directions]
        # lstm_output: [seq_len, batch_size, hidden_size]
        #print(x)
        #print(self.hidden)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        last_hidden_state = lstm_out[-1]

        if self.output_dim is None:
            if not self.output_sequence:
                return last_hidden_state
            else:
                return lstm_out.transpose(0, 1)

        if not self.output_sequence:
            return self.mlp(last_hidden_state)

        #at this point, we need to output a sequence, so we use the mlp to process the all sequences
        seq_len = lstm_out.shape[0]
        grid_size = int(math.sqrt(seq_len))
        batch_size = lstm_out.shape[1]

        outputs = lstm_out.view(seq_len*batch_size, self.hidden_dim)
        outputs = self.mlp(outputs)
        if type(outputs) == tuple:
            outputs = outputs[1]  # might want to keep the tuples, so they can be processed as well. not now.
        outputs = outputs.view(seq_len, batch_size)
        outputs = outputs.transpose(0, 1)   # batch_size, length_size
        outputs = outputs.contiguous().view(batch_size, grid_size, grid_size)
        return outputs


class NetGRAPHNew(SaveableNNModule):
    def __init__(self, name, output_activation=None, input_ch=2, num_output_channels=100):
        """
        :param output_activation: [None, F.softmax, F.sigmoid]
        """
        super(NetGRAPHNew, self).__init__()

        self.name = name
        self.output_activation = output_activation

        noch = num_output_channels

        self.conv1 = nn.Conv2d(in_channels=input_ch, out_channels=noch, kernel_size=3, padding=1)
        self.nullify_3d_corners(self.conv1.weight.data)  # set the corner weights to 0
        self.conv1.weight.register_hook(self.nullify_3d_corners)  # do the same for every gradient

        self.conv2 = nn.Conv2d(in_channels=noch, out_channels=noch, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=noch, out_channels=noch, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=noch, out_channels=noch, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=noch, out_channels=noch, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=noch, out_channels=1, kernel_size=1, padding=0)

    def forward(self, graph):
        # if x is a 2d list, convert it to a variable
        if type(graph) == list:
            # check if it's a list of tuples
            if graph.__len__() > 0 and type(graph[0]) == list \
                    and type(graph[0][0]) == tuple:
                # graph = [i[1] for i in graph]
                graph = [[j[1] for j in i] for i in graph]

            if graph.__len__() > 0 and type(graph[0]) == list:
                #concatenate all along cols
                graph = [[torch.unsqueeze(j, dim=2) for j in i] for i in graph]
                graph = [torch.cat(i, dim=2) for i in graph]

                #concatenate along rows
                graph = [torch.unsqueeze(a, dim=2) for a in graph]
                graph = torch.cat(graph, dim=2)
        elif type(graph) == tuple:
            graph = graph[1]

        speeds = torch.unsqueeze(graph[:, 0, :, :], dim=1)

        x = F.relu(self.conv1(graph))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        graph_values = x

        output = torch.cat((speeds, graph_values), dim=1)

        return torch.squeeze(graph_values), output

    @staticmethod
    def nullify_3d_corners(gradient):
        gradient[:, :, 0, 0] = 0
        gradient[:, :, 2, 2] = 0
        gradient[:, :, 0, 2] = 0
        gradient[:, :, 2, 0] = 0


def get_nn_from_params_dict(uf):
    new_nn = None
    if uf["type"] == "MLP":
        new_nn = NetMLP(uf["name"],
                        uf["input_dim"], uf["output_dim"], uf["output_activation"])
    elif uf["type"] == "CNN":
        new_nn = NetCNN(uf["name"], uf["input_dim"], uf["input_ch"])
    elif uf["type"] == "RNN":
        output_dim = uf["output_dim"] if "output_dim" in uf else None
        output_activation = uf["output_activation"] if "output_activation" in uf else None
        output_sequence = uf["output_sequence"] if "output_sequence" in uf else False
        new_nn = NetRNN(uf["name"], uf["input_dim"], uf["hidden_dim"],
                        output_dim=output_dim, output_activation=output_activation,
                        output_sequence=output_sequence)
    elif uf["type"] == "GCONVNew":
        new_nn = NetGRAPHNew(uf["name"], None, uf["input_dim"], num_output_channels=100)
    else:
        raise NotImplementedError()

    if "initialize_from" in uf and uf["initialize_from"] is not None:
        new_nn.load(uf["initialize_from"])

    if torch.cuda.is_available():
        new_nn.cuda()

    new_nn.params_dict = uf
    c_trainable_parameters = list(new_nn.parameters())

    return new_nn, c_trainable_parameters
