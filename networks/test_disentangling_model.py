'''File containing class for model to test disentangling of features'''
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.visualisation_utils as visualisation_utils
from .network_component import NetworkComponent

class TestDisentanglingModel(nn.Module):
    '''
        Train only a new AUPredictor on top of z2
    '''

    def __init__(self, cfg, args, is_train, save_path):

        super(TestDisentanglingModel, self).__init__()

        self.cfg = cfg
        self.args = args
        self.is_train = is_train

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.save_path = save_path

        # Encoder: takes in image and creates lower-dimensional representation.
        # Part or all of this representation is used by the AUPredictor
        self.encoder = NetworkComponent(self.cfg['Encoder'], self.args, name='Encoder')\
                       .to(self.args.device)

        self.encoder.load()

        # change number of input channels to size of z2
        self.z1_size = self.cfg['AUPredictor'][0]['channels']
        z2_size = self.encoder.output_size - self.z1_size
        self.cfg['AUPredictor'][0]['channels'] = z2_size

        # AUPredictor: takes in all or a subset of the embedding produced by the encoder
        # and predicts the action unit activations from them
        self.aupredictor = NetworkComponent(self.cfg['AUPredictor'], self.args,
                                            name='Test_Disentangling_AUPredictor').to(self.args.device)

        # Initialise Optimiser
        self.optimiser = torch.optim.Adam(self.aupredictor.parameters(),
                                          lr=self.aupredictor.blocks[0]['learning_rate'],
                                          betas=(self.aupredictor.blocks[0]['beta1'],
                                                 self.aupredictor.blocks[0]['beta2']))
        # Initialise Loss function
        if self.cfg['General'][2]['au_regression_loss_function'] == 'MSE':
            self.au_regression_loss_function = torch.nn.MSELoss()

        self.au_regression_loss = Variable(torch.zeros(1, device=self.args.device))

    def update_learning_rate(self, n_epoch):
        '''
            Update the learning rate based on n_epoch.
            Currently the learning rate simply decays linearly over the last
            few epochs; if more complicated behaviour is desired I might
            create a proper learning rate scheduler that update_learning_rate()
            simply calls .step() on.
        '''
        if self.cfg['General'][0]['learning_rate_scheduler'] == 'Linear':

            n_epochs_lr_decay = self.cfg['General'][0]['decay_lr_over_last_n_epochs']

            if self.cfg['General'][0]['max_epochs'] - n_epoch >= n_epochs_lr_decay:
                return

            # Calculate decay
            au_predictor_learning_rate_decay = self.cfg['AUPredictor'][0]['learning_rate'] \
                                        / n_epochs_lr_decay

            for param_group in self.optimiser.param_groups:

                param_group['lr'] -= au_predictor_learning_rate_decay
                print('Updated AUPredictor Learning Rate %f -> %f'
                      % (param_group['lr'] + au_predictor_learning_rate_decay,
                         param_group['lr']))

    def forward(self, batch):
        '''
            Run and calculate errors without making optimiser step.
            Return None, au_predictions for legacy reasons
        '''

        assert not self.is_train, \
            'Method TestDisentanglingModel.forward() called while in training mode'

        # Read in image and action unit activations
        image = Variable(batch[0]).to(device=self.args.device)
        action_units = Variable(batch[1]).to(device=self.args.device, dtype=torch.float)

        # Forward Pass
        encoded_features = self.encoder.forward(image)
        au_predictions = self.aupredictor.forward(
            encoded_features[:, self.z1_size:])

        self.au_regression_loss = self.au_regression_loss_function(au_predictions.squeeze(),
                                                                   action_units)

        return None, au_predictions

    def optimize_parameters(self, _, train_batch, log_results):
        '''
            Makes one forward run and backpropagate losses;
            update weights.
            Parameters:
                train_batch: Batch with training data, tuple (image, AU-activations)

            Returns:
                None (for legacy reasons), Action Unit Predictions and Gradient Flow Plot
                if log_results
                None, None if not log_results
        '''

        assert self.is_train, \
            'Method TestDisentanglingModel.optimize_parameters() called while not in training mode!'

        # Read in image and action unit activations
        image = Variable(train_batch[0].to(device=self.args.device))
        action_units = Variable(train_batch[1].to(device=self.args.device, dtype=torch.float))

        # Encoder: forward pass to create encoded_features
        encoded_features = self.encoder.forward(image)

        # AUPredictor: forward pass
        au_predictions = self.aupredictor.forward(
            encoded_features[:, self.z1_size:].detach())

        self.au_regression_loss = self.au_regression_loss_function(au_predictions.squeeze(),
                                                                   action_units)

        self.optimiser.zero_grad()
        self.au_regression_loss.backward()
        self.optimiser.step()

        if log_results:
            aupredictor_grad_flow_plot, _ = visualisation_utils.plot_gradient_flow(
                self.aupredictor.layers.named_parameters(),
                self.aupredictor.name)

            return None, au_predictions, \
                   {self.aupredictor.name + ' Gradient Flow': aupredictor_grad_flow_plot}

        return None, None, None

    def get_current_errors(self):
        '''
            Returns errors from latest batch;
            this method is used by different logging functions
        '''
        loss_dict = OrderedDict(
            [('Action Unit Regression Loss', self.au_regression_loss.item())])

        return loss_dict

    def save(self, n_epoch):
        '''
            Saves the new AUPredictor in self.save_path directory
        '''
        self.aupredictor.save(n_epoch, self.save_path)
