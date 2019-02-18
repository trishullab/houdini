import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from abc import abstractmethod
import json

from HOUDINI.Interpreter.NeuralModules import SaveableNNModule


class NetPNN(SaveableNNModule):
    def __init__(self, name, input_dim, input_ch, output_dim, output_activation=None, past_models=None):
        """
        :param output_activation: [None, F.softmax, F.sigmoid]
        :param past_models: a list of the previous models
        """
        super(NetPNN, self).__init__()

        self.name = name
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.past_models = past_models

        self.num_prev_models = 0 if past_models is None else past_models.__len__()

        self.layer_sizes = [32, 64, 1024]

        # CNN Layer 1
        self.conv1 = nn.Conv2d(input_ch, self.layer_sizes[0], kernel_size=5)
        conv1_output_dim = self.cnn_get_output_dim(input_dim, 5, stride=1, padding=0)
        pool1_output_dim = self.cnn_get_output_dim(conv1_output_dim, 2, stride=2, padding=0)

        # CNN Layer 2
        if self.num_prev_models > 0:
            self.scalar_2 = nn.Parameter(torch.from_numpy(np.ones(1, dtype=np.float32)))
            self.V_2 = nn.Conv2d(self.layer_sizes[0]*self.num_prev_models, self.layer_sizes[0], kernel_size=1)
            self.U_2 = nn.Conv2d(self.layer_sizes[0], self.layer_sizes[1], kernel_size=5, bias=False)

        self.conv2 = nn.Conv2d(self.layer_sizes[0], self.layer_sizes[1], kernel_size=5)
        conv2_output_dim = self.cnn_get_output_dim(pool1_output_dim, kernel_size=5, stride=1, padding=0)
        self.pool2_output_dim = self.cnn_get_output_dim(conv2_output_dim, 2, stride=2, padding=0)
        self.conv2_drop = nn.Dropout2d()

        # FC Layer 1
        self.fc1_input_dim = self.pool2_output_dim**2*self.layer_sizes[1]
        if self.num_prev_models > 0:
            self.scalar_3 = nn.Parameter(torch.from_numpy(np.ones(1, dtype=np.float32)))
            self.V_3 = nn.Conv2d(self.layer_sizes[1]*self.num_prev_models, self.layer_sizes[1], kernel_size=1)
            self.U_3 = nn.Linear(self.fc1_input_dim, self.layer_sizes[2], bias=False)

        self.fc1 = nn.Linear(self.fc1_input_dim, self.layer_sizes[2])
        self.bn1 = nn.BatchNorm1d(self.layer_sizes[2])

        # FC Layer 2
        if self.output_dim is not None:
            if self.num_prev_models > 0:
                self.scalar_4 = nn.Parameter(torch.from_numpy(np.ones(1, dtype=np.float32)))
                self.V_4 = nn.Linear(self.layer_sizes[2] * self.num_prev_models, self.layer_sizes[2])
                self.U_4 = nn.Linear(self.layer_sizes[2], output_dim, bias=False)
            self.fc2 = nn.Linear(self.layer_sizes[2], output_dim)

    def cnn_get_output_dim(self, w1, kernel_size, stride, padding=0):
        w2 = (w1 - kernel_size + 2*padding) // stride + 1
        return w2

    def get_activations_for_rnn(self, x_list_tensor):

        x_list = torch.split(x_list_tensor, split_size=1, dim=1)
        x_list = [torch.squeeze(ii, dim=1) for ii in x_list]

        if self.num_prev_models > 0:
            prev_model = self.past_models[-1]
            # prev_activations = {"recogn": [[[]_layer]_model]_item, "rnn": [[]_layer]_model}
            if type(prev_model) == NetPNN:
                prev_activations = prev_model.get_activations_for_rnn(x_list_tensor)
            else:
                prev_activations = prev_model(x_list_tensor, return_activations=True)
            prev_activations["rnn"].append([])
        else:
            prev_activations = {"recogn": [], "rnn": [[]]}

        for x_idx, x in enumerate(x_list):
            if self.num_prev_models > 0:
                # generate all the prev activations for the current item
                prev_activations_for_c_item = prev_activations["recogn"][x_idx]
                new_activations = self.forward(x, return_activations=True, prev_activations=prev_activations_for_c_item)
                prev_activations["recogn"][x_idx] = new_activations
            else:
                new_activations = self.forward(x, return_activations=True, prev_activations=None)
                prev_activations["recogn"].append(new_activations)

        return prev_activations
        """
        list_prev_activations = [self.forward(x, return_activations=True) for x in x_list]
        activations_for_current_model = {"recogn:": list_prev_activations[-1], "rnn": None}
        list_prev_activations[-1] = activations_for_current_model
        """

    def forward(self, x, return_activations=False, prev_activations=None):
        if self.num_prev_models > 0:
            if prev_activations is None:
                prev_model = self.past_models[-1]
                # prev_activations = [[]_layer]_model]
                if type(prev_model) == NetPNN:
                    prev_activations = prev_model(x, return_activations=True)
                else:
                    prev_activations = prev_model.recogniser(x, return_activations=True)
                # prev_activations = [[], ..., []_layers ]_models
            prev_activations = prev_activations[-7:]
            # print("Prev_acivations.__length__(): {}".format(prev_activations.__len__()))
        activations = []
        # CNN Layer 1:
        activation1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        activations.append(activation1)

        # CNN Layer 2:
        #main_column's logits:
        mcl2 = self.conv2(activation1)
        if self.num_prev_models > 0:
            # d = [m[0] for m in prev_activations]
            anterior_features = torch.cat([m[0] for m in prev_activations], 1)
            projection = F.relu(self.V_2(self.scalar_2*anterior_features))
            mcl2 += self.U_2(projection)

        activation2 = F.relu(F.max_pool2d(self.conv2_drop(mcl2), 2))
        activations.append(activation2)

        # FC Layer 1
        # main column's logits:
        x = activation2.view(-1, self.fc1_input_dim)
        mcl3 = self.fc1(x)
        if self.num_prev_models > 0:
            anterior_features = torch.cat([m[1] for m in prev_activations], 1)
            projection = F.relu(self.V_3(self.scalar_3 * anterior_features))
            projection_reshaped = projection.view(-1, self.fc1_input_dim)
            mcl3 += self.U_3(projection_reshaped)

        activation3 = F.dropout(self.bn1(F.relu(mcl3)), training=self.training)
        activations.append(activation3)

        # FC Layer 2
        # main column's logits:
        mcl4 = self.fc2(activation3)
        if self.num_prev_models > 0:
            anterior_features = torch.cat([m[2] for m in prev_activations], -1)
            projection = F.relu(self.V_4(self.scalar_4 * anterior_features))
            mcl4 += self.U_4(projection)

        activation4 = mcl4 if self.output_activation is None else self.output_activation(mcl4)
        activations.append(activation4)

        if return_activations:
            if self.num_prev_models > 0:
                return prev_activations + [activations]
            else:
                return [activations]
        else:
            return mcl4, activation4


class NetPNN_RNN(SaveableNNModule):
    def __init__(self, name, input_dim, input_ch, recogniser_output_dim, recogniser_output_activation,
                 output_dim, output_activation=None, past_models=None):
        """
        :param output_activation: [None, F.softmax, F.sigmoid]
        :param past_models: a list of the previous models
        """
        super(NetPNN_RNN, self).__init__()

        if past_models is not None:
            # print(past_models.__len__())
            past_models = past_models[-7:]
            # print(past_models.__len__())
        
        self.name = name
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.past_models = past_models

        self.hidden_dim = 100

        self.num_prev_models = 0 if past_models is None else past_models.__len__()

        self.recogniser = NetPNN("{}_recogniser".format(name), input_dim, input_ch,
                                 recogniser_output_dim, recogniser_output_activation, past_models)

        # RNN Layer
        self.hidden = None  # a placeholder, used for the hidden state of the lstm
        if self.num_prev_models > 0:
            self.scalar_5 = nn.Parameter(torch.from_numpy(np.ones(1, dtype=np.float32)))
            self.V_5 = nn.Linear(recogniser_output_dim * self.num_prev_models, recogniser_output_dim)
            # self.U_5 = nn.Linear(300, output_dim, bias=False)
            self.lstm = nn.LSTM(input_size=recogniser_output_dim*2, hidden_size=self.hidden_dim)
        else:
            self.lstm = nn.LSTM(input_size=recogniser_output_dim, hidden_size=self.hidden_dim)

        num_prev_rnns = 0
        if past_models is not None:
            for i in past_models:
                if type(i) == NetPNN_RNN:
                    num_prev_rnns += 1

        # FC Layer 1
        if num_prev_rnns > 0:
            self.scalar_6 = nn.Parameter(torch.from_numpy(np.ones(1, dtype=np.float32)))
            self.V_6 = nn.Linear(self.hidden_dim*num_prev_rnns, self.hidden_dim)
            self.U_6 = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)

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

    def forward(self, x_list, return_activations=False):

        prev_activations = self.recogniser.get_activations_for_rnn(x_list)

        main_recogniser_activations = [item[-1][3] for item in prev_activations["recogn"]]
        main_recogniser_activations = [torch.unsqueeze(a, dim=1) for a in main_recogniser_activations]
        main_recogniser_activations = torch.cat(main_recogniser_activations, dim=1)

        if self.num_prev_models > 0:
            prev_recogniser_activations = []
            for model_index in range(self.num_prev_models):
                # turn a model's input for a list to [batch_size, length, 1]
                outputs_from_model_i = [item[model_index][3] for item in prev_activations["recogn"]]
                outputs_from_model_i = [torch.unsqueeze(a, dim=1) for a in outputs_from_model_i]
                outputs_from_model_i = torch.cat(outputs_from_model_i, dim=1)
                prev_recogniser_activations.append(outputs_from_model_i)

            prev_recogniser_activations = torch.cat(prev_recogniser_activations, dim=2)
            anterior_features = F.relu(self.V_5(self.scalar_5*prev_recogniser_activations))
            main_recogniser_activations = torch.cat([main_recogniser_activations, anterior_features], dim=2)

        batch_size = x_list.data.shape[0]
        self.reset_hidden_state(batch_size)
        # from [batch_size, list_size, items] to [list_size, batch_size, items]
        #print("main_recogniser_activations.shape: {}".format(main_recogniser_activations.shape))
        main_recogniser_activations = torch.transpose(main_recogniser_activations, 0, 1)
        #print("main_recogniser_activations.shape: {}".format(main_recogniser_activations.shape))

        lstm_out, self.hidden = self.lstm(main_recogniser_activations, self.hidden)

        #print("lstm_out.shape: {}".format(lstm_out.shape))

        last_hidden_state = lstm_out[-1]

        activations = [last_hidden_state]
        #print("last_hidden_state.shape: {}".format(last_hidden_state.shape))

        # main column's logits:
        mcl6 = self.fc1(last_hidden_state)
        if self.num_prev_models > 0:
            prev_activations["rnn"] = prev_activations["rnn"][-8:]
            prev_rnn_activations = [m[0] for m in prev_activations["rnn"][:-1] if m.__len__() > 0]
            if prev_rnn_activations.__len__() > 0:
                anterior_features = torch.cat(prev_rnn_activations, -1)
                projection = F.relu(self.V_6(self.scalar_6 * anterior_features))
                mcl6 += self.U_6(projection)

        activation6 = mcl6 if self.output_activation is None else self.output_activation(mcl6)
        activations.append(activation6)

        prev_activations["rnn"][-1] = activations

        if return_activations:
            return prev_activations
            # if self.num_prev_models > 0:
            #    return prev_activations
        else:
            return mcl6, activation6
