''' File containing the Model class '''

import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.visualisation_utils as visualisation_utils
import utils.metric_utils as metric_utils
from .network_component import NetworkComponent

class Model(nn.Module):
    '''
        Class for the whole network with all parts
    '''

    def __init__(self, cfg, args, is_train, save_path):
        super(Model, self).__init__()

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

        # Decoder: takes in embedding produced by the encoder and reconstructs the
        # original image
        self.decoder = NetworkComponent(self.cfg['Decoder'], self.args, name='Decoder')\
                       .to(self.args.device)

        # Discriminator: used to minimize mutual information between parts of the embedding
        # produced by the encoder
        self.discriminator = NetworkComponent(self.cfg['Discriminator'], self.args,
                                              name='Discriminator').to(self.args.device)

        # Image Discriminator: used for providing a loss signal for reconstructing the image
        self.image_discriminator = NetworkComponent(self.cfg['Image Discriminator'], self.args,
                                                    name='Image Discriminator').to(self.args.device)

        # AUPredictor: takes in all or a subset of the embedding produced by the encoder
        # and predicts the action unit activations from them
        self.aupredictor = NetworkComponent(self.cfg['AUPredictor'], self.args,
                                            name='AUPredictor').to(self.args.device)

        # Flag whether to use Discriminator or not
        # Set frequency of training the main network
        if self.encoder.output_size == self.aupredictor.input_size:
            self.use_discriminator = False
        else:
            self.use_discriminator = True

        # This is always what is specified because we always use the Image Discriminator
        self.train_main_net_every_n_batches = \
                self.cfg['General'][0]['train_main_net_every_n_batches']

        # initialise optimiser and other variables
        if self.is_train:
            self.initialise_optimisers()

        # load networks and optimizers
        if not self.is_train or self.args.load_epoch > 0:
            self.load()

        # Initialise loss functions
        if self.cfg['General'][2]['au_regression_loss_function'] == 'MSE':
            self.au_regression_loss_function = torch.nn.MSELoss()
        elif self.cfg['General'][2]['au_regression_loss_function'] == 'BCEWithLogits':
            self.au_regression_loss_function = torch.nn.BCEWithLogitsLoss()

        if self.cfg['General'][2]['reconstruction_loss_function'] == 'L1':
            self.reconstruction_loss_function = torch.nn.L1Loss()

        # Loss for Generator:
        self.au_regression_loss = Variable(torch.zeros(1, device=self.args.device))
        self.reconstruction_loss = Variable(torch.zeros(1, device=self.args.device))

        # Loss for Image Discriminator:
        self.real_image_loss = Variable(torch.zeros(1, device=self.args.device))
        self.gen_image_loss = Variable(torch.zeros(1, device=self.args.device))
        self.img_discr_gradient_penalty = Variable(torch.zeros(1, device=self.args.device))

        # Loss for Discriminator
        self.real_pairs_loss = Variable(torch.zeros(1, device=self.args.device))
        self.shuffled_pairs_loss = Variable(torch.zeros(1, device=self.args.device))
        self.discr_gradient_penalty = Variable(torch.zeros(1, device=self.args.device))

        # Initialise scalars
        self.z1_z2_similarity_measure = metric_utils.x_cov # Cross Covariance (XCov)
        self.z1_z2_similarity_value = torch.zeros(1, device=self.args.device)

        # Average Variance =^= Mean of Diagonal of Covariance matrix
        self.z2_z2_variability_measure = metric_utils.average_variance
        self.z2_z2_variability_value = torch.zeros(1, device=self.args.device)

        self.active_au_threshold_pred = 0.5 # only needed to convert network output to boolean
        self.active_au_threshold_gt = 0.5 # only needed to convert ground truths to boolean
        self.true_positives = torch.zeros(self.aupredictor.output_size, device=self.args.device)
        self.true_negatives = torch.zeros(self.aupredictor.output_size, device=self.args.device)
        self.false_positives = torch.zeros(self.aupredictor.output_size, device=self.args.device)
        self.false_negatives = torch.zeros(self.aupredictor.output_size, device=self.args.device)

        # I need to save those to compute 2AFC value
        self.au_predictions = None
        self.au_ground_truths = None

    def initialise_optimisers(self):
        '''
            Method to initialise optimisers for different network parts
        '''

        # Initialise Optimiser for Encoder/AUPredictor/Decoder

        # Collect parameters for Encoder
        encoder_params = []
        for param in self.encoder.parameters():
            if param.requires_grad:
                encoder_params.append(param)

        # Collect parameters for AUPredictor
        aupredictor_params = []
        for param in self.aupredictor.parameters():
            if param.requires_grad:
                aupredictor_params.append(param)

        # Collect parameters for Decoder
        decoder_params = []
        for param in self.decoder.parameters():
            if param.requires_grad:
                decoder_params.append(param)


        main_params = [
            {'network_name': 'Encoder',
             'params': encoder_params,
             'lr':     self.encoder.blocks[0]['learning_rate'],
             'betas':  (self.encoder.blocks[0]['beta1'],
                        self.encoder.blocks[0]['beta2'])},
            {'network_name': 'AUPredictor',
             'params': aupredictor_params,
             'lr':     self.aupredictor.blocks[0]['learning_rate'],
             'betas':  (self.aupredictor.blocks[0]['beta1'],
                        self.aupredictor.blocks[0]['beta2'])},
            {'network_name': 'Decoder',
             'params': decoder_params,
             'lr':     self.decoder.blocks[0]['learning_rate'],
             'betas':  (self.decoder.blocks[0]['beta1'],
                        self.decoder.blocks[0]['beta2'])},
        ]

        self.optimiser_main = torch.optim.Adam(main_params)

        # Collect parameters for Image Discriminator
        img_discriminator_params = []
        for param in self.image_discriminator.parameters():
            if param.requires_grad:
                img_discriminator_params.append(param)

        # Initialise Optimiser for Image Discriminator
        self.optimiser_img_disc = \
                            torch.optim.Adam(img_discriminator_params,
                                             lr=self.image_discriminator.blocks[0]['learning_rate'],
                                             betas=(self.image_discriminator.blocks[0]['beta1'],
                                                    self.image_discriminator.blocks[0]['beta2']))

        # Collect parameters for Discriminator
        discriminator_params = []
        for param in self.discriminator.parameters():
            if param.requires_grad:
                discriminator_params.append(param)

        # Initialise Optimiser for Discriminator
        if self.use_discriminator:
            self.optimiser_disc = torch.optim.Adam(discriminator_params,
                                                   lr=self.discriminator.blocks[0]['learning_rate'],
                                                   betas=(self.discriminator.blocks[0]['beta1'],
                                                          self.discriminator.blocks[0]['beta2']))

    def load(self):
        '''
            Load networks from args.load_dir
        '''

        # load Encoder
        self.encoder.load()

        # load AUPredictor
        self.aupredictor.load()

        if self.is_train:
            # load Decoder
            self.decoder.load()

            # load Optimiser for Encoder/AUPredictor/Decoder
            optim_main_filename = 'optim_main_epoch_%s.pth' % self.args.load_epoch
            optim_main_filepath = os.path.join(self.args.load_dir, optim_main_filename)

            assert os.path.exists(optim_main_filepath), \
                'Requested Optimiser Main File does not exist: %s' % optim_main_filepath

            self.optimiser_main.load_state_dict(torch.load(optim_main_filepath))
            print('loaded Optimiser Main: %s' % optim_main_filepath)

            # Set correct learning rates
            for param_group in self.optimiser_main.param_groups:
                if param_group['network_name'] == 'Encoder':
                    param_group['lr'] = self.encoder.blocks[0]['learning_rate']
                    print('Set Encoder learning rate to %f'
                          % param_group['lr'])
                elif param_group['network_name'] == 'AUPredictor':
                    param_group['lr'] = self.aupredictor.blocks[0]['learning_rate']
                    print('Set AUPredictor learning rate to %f'
                          % param_group['lr'])
                elif param_group['network_name'] == 'Decoder':
                    param_group['lr'] = self.decoder.blocks[0]['learning_rate']
                    print('Set Decoder learning rate to %f'
                          % param_group['lr'])

            # load Image Discriminator
            self.image_discriminator.load()

            # load Optimiser for Image Discriminator
            optim_img_disc_filename = 'optim_img_disc_epoch_%s.pth' % self.args.load_epoch
            optim_img_disc_filepath = os.path.join(self.args.load_dir, optim_img_disc_filename)

            assert os.path.exists(optim_img_disc_filepath), \
                'Requested Optimiser Image Discr File does not exist: %s' % optim_img_disc_filepath

            self.optimiser_img_disc.load_state_dict(torch.load(optim_img_disc_filepath))
            print('loaded Optimiser Image Discr: %s' % optim_img_disc_filepath)

            # Set correct learning rate
            for param_group in self.optimiser_img_disc.param_groups:
                param_group['lr'] = self.image_discriminator.blocks[0]['learning_rate']
                print('Set Image Discriminator learning rate to %f' % param_group['lr'])


            # load Model and Optimiser for Discriminator
            if self.use_discriminator:

                # Load Model
                self.discriminator.load()

                # Load Optimiser
                optim_disc_filename = 'optim_disc_epoch_%s.pth' % self.args.load_epoch
                optim_disc_filepath = os.path.join(self.args.load_dir, optim_disc_filename)

                assert os.path.exists(optim_disc_filepath), \
                    'Requested Optimiser Discr File does not exist: %s' % optim_disc_filepath

                self.optimiser_disc.load_state_dict(torch.load(optim_disc_filepath))
                print('loaded Optimiser Discr: %s' % optim_disc_filepath)

                # Set correct learning rate
                for param_group in self.optimiser_disc.param_groups:
                    param_group['lr'] = self.discriminator.blocks[0]['learning_rate']
                    print('Set Discriminator learning rate to %f'
                          % param_group['lr'])

    def save(self, n_epoch):
        '''
            Saves the model weights in self.save_path directory
        '''
        # Save Encoder
        self.encoder.save(n_epoch, self.save_path)

        # Save AUPredictor
        self.aupredictor.save(n_epoch, self.save_path)

        # Save Decoder
        self.decoder.save(n_epoch, self.save_path)

        # Save Optimiser for Encoder/AUPredictor/Decoder
        optim_main_filename = 'optim_main_epoch_%s.pth' % n_epoch
        optim_main_filepath = os.path.join(self.save_path, optim_main_filename)

        torch.save(self.optimiser_main.state_dict(), optim_main_filepath)

        print('Saved Optimiser for Main Network: %s' % optim_main_filepath)

        # Save Image Discriminator
        self.image_discriminator.save(n_epoch, self.save_path)

        # Save Optimiser for Image Discriminator
        optim_img_disc_filename = 'optim_img_disc_epoch_%s.pth' % n_epoch
        optim_img_disc_filepath = os.path.join(self.save_path, optim_img_disc_filename)

        torch.save(self.optimiser_img_disc.state_dict(), optim_img_disc_filepath)

        print('Saved Optimiser for Image Discriminator: %s' % optim_img_disc_filepath)

        # Save Model and Optimiser for Discriminator
        if self.use_discriminator:
            # Save Discriminator Model
            self.discriminator.save(n_epoch, self.save_path)

            # Save Discriminator Optimiser
            optim_disc_filename = 'optim_disc_epoch_%s.pth' % n_epoch
            optim_disc_filepath = os.path.join(self.save_path, optim_disc_filename)

            torch.save(self.optimiser_disc.state_dict(), optim_disc_filepath)

            print('Saved Optimiser for Discriminator: %s' % optim_disc_filepath)

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

            # Calculate decays

            encoder_learning_rate_decay = self.cfg['Encoder'][0]['learning_rate'] \
                                        / n_epochs_lr_decay
            au_predictor_learning_rate_decay = self.cfg['AUPredictor'][0]['learning_rate'] \
                                        / n_epochs_lr_decay
            decoder_learning_rate_decay = self.cfg['Decoder'][0]['learning_rate'] \
                                        / n_epochs_lr_decay

            image_discriminator_learning_rate_decay = \
                            self.cfg['Image Discriminator'][0]['learning_rate'] / n_epochs_lr_decay

            if self.use_discriminator:
                discriminator_learning_rate_decay = self.cfg['Discriminator'][0]['learning_rate'] \
                                            / n_epochs_lr_decay

                # Apply Decays
                for param_group in self.optimiser_disc.param_groups:
                    param_group['lr'] -= discriminator_learning_rate_decay
                    print('Updated Discriminator Learning Rate %f -> %f'
                          % (param_group['lr'] + discriminator_learning_rate_decay,
                             param_group['lr']))

            for param_group in self.optimiser_main.param_groups:
                if param_group['network_name'] == 'Encoder':
                    param_group['lr'] -= encoder_learning_rate_decay
                    print('Updated Encoder Learning Rate %f -> %f'
                          % (param_group['lr'] + encoder_learning_rate_decay, param_group['lr']))
                elif param_group['network_name'] == 'AUPredictor':
                    param_group['lr'] -= au_predictor_learning_rate_decay
                    print('Updated AUPredictor Learning Rate %f -> %f'
                          % (param_group['lr'] + au_predictor_learning_rate_decay,
                             param_group['lr']))
                elif param_group['network_name'] == 'Decoder':
                    param_group['lr'] -= decoder_learning_rate_decay
                    print('Updated Decoder Learning Rate %f -> %f'
                          % (param_group['lr'] + decoder_learning_rate_decay, param_group['lr']))

            for param_group in self.optimiser_img_disc.param_groups:
                param_group['lr'] -= image_discriminator_learning_rate_decay
                print('Updated Image Discriminator Learning Rate %f -> %f'
                      % (param_group['lr'] + image_discriminator_learning_rate_decay,
                         param_group['lr']))

    def forward(self, batch):
        '''
            Only run main network and calculate errors
            Returns reconstructed image and au_predictions
            Batch is of the form (image, action_units); no noisy_image
        '''

        assert not self.is_train, 'Method Model.forward() called while in training mode'

        # Read in image and action unit activations
        image = Variable(batch[0]).to(device=self.args.device)
        action_units = Variable(batch[1]).to(device=self.args.device, dtype=torch.float)

        # Forward Pass
        encoded_features = self.encoder.forward(image)
        reconstructed_image = self.decoder.forward(encoded_features)
        au_predictions = self.aupredictor.forward(
            encoded_features[:, 0:self.aupredictor.input_size]).squeeze()

        self.real_image_loss = torch.mean(self.image_discriminator.forward(image))
        self.gen_image_loss = torch.mean(self.image_discriminator.forward(reconstructed_image))

        self.au_regression_loss = self.au_regression_loss_function(au_predictions,
                                                                   action_units)
        self.reconstruction_loss = self.reconstruction_loss_function(image,
                                                                     reconstructed_image)

        # Calculate scalars
        if self.use_discriminator:
            self.z1_z2_similarity_value = self.z1_z2_similarity_measure(
                encoded_features[:, :self.aupredictor.input_size],
                encoded_features[:, self.aupredictor.input_size:])

            self.z2_z2_variability_value = self.z2_z2_variability_measure(
                encoded_features[:, self.aupredictor.input_size:])

        # This will still work if we use binary target values
        if self.cfg['General'][1]['regress_au_intensities']:
            au_predictions = torch.nn.Sigmoid()(au_predictions)

        self.true_positives = (au_predictions > self.active_au_threshold_pred) \
                            & (action_units > self.active_au_threshold_gt)
        self.true_positives = torch.sum(self.true_positives, 0, dtype=torch.float)

        self.true_negatives = (au_predictions <= self.active_au_threshold_pred) \
                            & (action_units <= self.active_au_threshold_gt)
        self.true_negatives = torch.sum(self.true_negatives, 0, dtype=torch.float)

        self.false_positives = (au_predictions > self.active_au_threshold_pred) \
                             & (action_units <= self.active_au_threshold_gt)
        self.false_positives = torch.sum(self.false_positives, 0, dtype=torch.float)

        self.false_negatives = (au_predictions <= self.active_au_threshold_pred) \
                             & (action_units > self.active_au_threshold_gt)
        self.false_negatives = torch.sum(self.false_negatives, 0, dtype=torch.float)

        self.au_predictions = au_predictions
        self.au_ground_truths = action_units > self.active_au_threshold_gt

        return reconstructed_image, au_predictions

    def optimize_parameters(self, batch_index, train_batch, log_results):
        '''
            Makes one forward run and backpropagate losses;
            update weights.
            Parameters:
                train_batch: Batch with training data, tuple (noisy_image, image, AU-activations)

            Returns:
                reconstructed image and Action Unit Predictions if log_results
                None, None if not log_results
        '''

        assert self.is_train, \
            'Method Model.optimize_parameters() called while not in training mode!'

        # Read in noisy_image and action unit activations
        noisy_image = Variable(train_batch[0].to(device=self.args.device))
        image = Variable(train_batch[1].to(device=self.args.device))
        action_units = Variable(train_batch[2].to(device=self.args.device, dtype=torch.float))

        # decide if we train main network
        train_main_network = ((batch_index+1) % self.train_main_net_every_n_batches == 0) \
                             or log_results

        # Forward Pass
        encoded_features = self.encoder.forward(noisy_image)
        reconstructed_image = self.decoder.forward(encoded_features)

        # run Image Discriminator
        img_disc_real_image_output = self.image_discriminator.forward(image)
        img_disc_gen_image_output = self.image_discriminator.forward(reconstructed_image.detach())

        # Calculate loss and update weights
        self.optimiser_img_disc.zero_grad()

        self.real_image_loss = torch.mean(img_disc_real_image_output)
        self.gen_image_loss = torch.mean(img_disc_gen_image_output)

        self.calculate_img_discriminator_gradient_penalty(image, reconstructed_image.detach())

        img_discriminator_loss = self.gen_image_loss - self.real_image_loss \
          + self.cfg['General'][2]['gradient_loss_lambda_img_discr']*self.img_discr_gradient_penalty

        img_discriminator_loss.backward()
        self.optimiser_img_disc.step()

        # if we use both z1 and z2, run the discriminator
        if self.use_discriminator:
            shuffled_encoding = self.shuffle_encoding(encoded_features.clone().detach())

            # Discriminator(real_pairs) --> (z1, z2) pairs generated by Encoder
            disc_real_pairs_output = self.discriminator.forward(encoded_features.detach())

            # Discriminator(shuffled_pairs) --> (z1, z2) shuffled
            disc_shuffled_pairs_output = self.discriminator.forward(shuffled_encoding)


            # Calculate loss and update weights
            self.optimiser_disc.zero_grad()

            self.real_pairs_loss = torch.mean(disc_real_pairs_output)
            self.shuffled_pairs_loss = torch.mean(disc_shuffled_pairs_output)

            self.calculate_discriminator_gradient_penalty(shuffled_encoding,
                                                          encoded_features.detach())

            discriminator_loss = self.real_pairs_loss - self.shuffled_pairs_loss \
                + self.cfg['General'][2]['gradient_loss_lambda_discr'] * self.discr_gradient_penalty

            discriminator_loss.backward()
            self.optimiser_disc.step()

        if train_main_network:

            # Recalculate Discriminator and Image Discriminator Outputs
            img_disc_gen_image_output = self.image_discriminator.forward(reconstructed_image)
            self.gen_image_loss = torch.mean(img_disc_gen_image_output)

            if self.use_discriminator:
                disc_real_pairs_output = self.discriminator.forward(encoded_features)
            else:
                disc_real_pairs_output = torch.zeros(1, device=self.args.device)                

            self.real_pairs_loss = torch.mean(disc_real_pairs_output)

            au_predictions = self.aupredictor.forward(
                encoded_features[:, 0:self.aupredictor.input_size]).squeeze()

            self.au_regression_loss = self.au_regression_loss_function(au_predictions,
                                                                       action_units)
            self.reconstruction_loss = self.reconstruction_loss_function(image,
                                                                         reconstructed_image)

            main_network_loss = self.au_regression_loss \
                + self.cfg['General'][2]['reconstruction_loss_lambda'] * self.reconstruction_loss \
                - self.cfg['General'][2]['img_discriminator_loss_lambda'] * self.gen_image_loss \
                - self.cfg['General'][2]['discriminator_loss_lambda'] * self.real_pairs_loss

            self.optimiser_main.zero_grad()
            main_network_loss.backward()
            self.optimiser_main.step()

            if log_results:

                # Calculate scalars
                if self.use_discriminator:
                    self.z1_z2_similarity_value = self.z1_z2_similarity_measure(
                        encoded_features[:, :self.aupredictor.input_size],
                        encoded_features[:, self.aupredictor.input_size:])

                    self.z2_z2_variability_value = self.z2_z2_variability_measure(
                        encoded_features[:, self.aupredictor.input_size:])

                # This will still work if we use binary target values
                if self.cfg['General'][1]['regress_au_intensities']:
                    au_predictions = torch.nn.Sigmoid()(au_predictions)

                self.true_positives = (au_predictions > self.active_au_threshold_pred) \
                                    & (action_units > self.active_au_threshold_gt)
                self.true_positives = torch.sum(self.true_positives, 0, dtype=torch.float)

                self.true_negatives = (au_predictions <= self.active_au_threshold_pred) \
                                    & (action_units <= self.active_au_threshold_gt)
                self.true_negatives = torch.sum(self.true_negatives, 0, dtype=torch.float)

                self.false_positives = (au_predictions > self.active_au_threshold_pred) \
                                     & (action_units <= self.active_au_threshold_gt)
                self.false_positives = torch.sum(self.false_positives, 0, dtype=torch.float)

                self.false_negatives = (au_predictions <= self.active_au_threshold_pred) \
                                     & (action_units > self.active_au_threshold_gt)
                self.false_negatives = torch.sum(self.false_negatives, 0, dtype=torch.float)

                self.au_predictions = au_predictions
                self.au_ground_truths = action_units > self.active_au_threshold_gt

                # Calculate Gradient Flow Plots
                encoder_grad_flow_plot, _ = visualisation_utils.plot_gradient_flow(
                    self.encoder.layers.named_parameters(),
                    self.encoder.name)

                aupredictor_grad_flow_plot, _ = visualisation_utils.plot_gradient_flow(
                    self.aupredictor.layers.named_parameters(),
                    self.aupredictor.name)

                decoder_grad_flow_plot, _ = visualisation_utils.plot_gradient_flow(
                    self.decoder.layers.named_parameters(),
                    self.decoder.name)

                gradient_flow_dict = \
                        {self.encoder.name + ' Gradient Flow': encoder_grad_flow_plot,
                         self.aupredictor.name + ' Gradient Flow': aupredictor_grad_flow_plot,
                         self.decoder.name + ' Gradient Flow': decoder_grad_flow_plot}

                if self.use_discriminator:

                    discriminator_grad_flow_plot, _ = visualisation_utils.plot_gradient_flow(
                        self.discriminator.layers.named_parameters(),
                        self.discriminator.name)

                    gradient_flow_dict[self.discriminator.name + ' Gradient Flow'] = \
                                       discriminator_grad_flow_plot


                return reconstructed_image, au_predictions, gradient_flow_dict

        return None, None, None

    def shuffle_encoding(self, encoded_features):
        '''
            Takes in a batch of embeddings produced by the encoder, and shuffles
            the z1 and z2 between batches so that z1 and z2 are independent
        '''
        # Read out z1 and z2
        z_1 = encoded_features[:, :self.aupredictor.input_size]
        z_2 = encoded_features[:, self.aupredictor.input_size:]

        # Move z2 elements by one index, because the dataloader shuffles
        # the batches, this corresponds to shuffling the (z1, z2) pairs
        inds = [ind for ind in range(1, z_2.size()[0])] + [0]
        z_2 = z_2[inds]

        # Merge z1 and z2 back together
        shuffled_encoding = torch.cat((z_1, z_2), dim=1)

        return shuffled_encoding.detach()

    def calculate_discriminator_gradient_penalty(self, shuffled_encoding, real_encoding):
        '''
            Calculates the gradient penalty (without regularization parameter lambda)
            for the discriminator and saves result in self.discr_gradient_penalty
        '''

        alpha = torch.rand((shuffled_encoding.size()[0], 1), device=self.args.device)
        alpha = alpha[:, :, None, None].expand_as(shuffled_encoding)

        # Interpolate between real and shuffled encoding
        interpolated = (alpha * shuffled_encoding) + ((1 - alpha) * real_encoding)
        interpolated = Variable(interpolated.to(self.args.device), requires_grad=True)

        # Pass interpolated values through discriminator
        discr_interpolated = self.discriminator.forward(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
                        outputs=discr_interpolated,
                        inputs=interpolated,
                        grad_outputs=torch.ones(discr_interpolated.size(), device=self.args.device),
                        retain_graph=True,
                        create_graph=True,
                        only_inputs=True)[0]

        self.discr_gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    def calculate_img_discriminator_gradient_penalty(self, real_image, gen_image):
        '''
            Calculates the gradient penalty (without regularization parameter lambda)
            for the image discriminator and saves result in self.img_discr_gradient_penalty
        '''

        alpha = torch.rand((real_image.size()[0], 1), device=self.args.device)
        alpha = alpha[:, :, None, None].expand_as(real_image)

        # Interpolate between real and shuffled encoding
        interpolated = (alpha * real_image) + ((1 - alpha) * gen_image)
        interpolated = Variable(interpolated.to(self.args.device), requires_grad=True)

        # Pass interpolated values through discriminator
        img_discr_interpolated = self.image_discriminator.forward(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=img_discr_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(img_discr_interpolated.size(), device=self.args.device),
            retain_graph=True,
            create_graph=True,
            only_inputs=True)[0]

        self.img_discr_gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    def get_current_errors(self):
        '''
            Returns errors from latest batch in an OrderedDict;
            this method is used by different logging functions
        '''
        loss_dict = OrderedDict(
            [('Action Unit Regression Loss', self.au_regression_loss.item()),
             ('Reconstruction Loss', self.reconstruction_loss.item()),
             ('Real (z1, z2) Pairs Discriminator Loss', self.real_pairs_loss.item()),
             ('Shuffled (z1, z2) Pairs Discriminator Loss', self.shuffled_pairs_loss.item()),
             ('Lower bound on Earth Mover Distance for Embeddings',
              self.shuffled_pairs_loss.item() - self.real_pairs_loss.item()),
             ('Image Discriminator Loss for Real Image', self.real_image_loss.item()),
             ('Image Discriminator Loss for Generated Image', self.gen_image_loss.item()),
             ('Lower bound on Earth Mover Distance for Images',
              self.real_image_loss.item() - self.gen_image_loss.item()),
             ('Discriminator Gradient Penalty', self.discr_gradient_penalty.item()),
             ('Image Discriminator Gradient Penalty', self.img_discr_gradient_penalty.item())])

        return loss_dict

    def get_current_scalars(self):
        '''
            Returns other metrics and scalars besides the errors from latest batch
            in an OrderedDict;
            this method is used by different logging functions
        '''

        if self.is_train:

            # Calculate F1 scores
            f1_denominator = 2 * self.true_positives + self.false_negatives \
                           + self.false_positives
            f1_denominator += torch.finfo(torch.float).tiny
            individual_f1_scores = 2 * self.true_positives / f1_denominator

            all_tp = torch.sum(self.true_positives)
            all_fn = torch.sum(self.false_negatives)
            all_fp = torch.sum(self.false_positives) + torch.finfo(torch.float).tiny

            average_f1_score = 2 * all_tp / (2 * all_tp + all_fn + all_fp)

            f1_scores = {'F1 Score for AU1': individual_f1_scores[0].item(),
                         'F1 Score for AU2': individual_f1_scores[1].item(),
                         'F1 Score for AU4': individual_f1_scores[2].item(),
                         'F1 Score for AU5': individual_f1_scores[3].item(),
                         'F1 Score for AU6': individual_f1_scores[4].item(),
                         'F1 Score for AU9': individual_f1_scores[5].item(),
                         'F1 Score for AU12': individual_f1_scores[6].item(),
                         'F1 Score for AU15': individual_f1_scores[7].item(),
                         'F1 Score for AU17': individual_f1_scores[8].item(),
                         'F1 Score for AU20': individual_f1_scores[9].item(),
                         'F1 Score for AU25': individual_f1_scores[10].item(),
                         'F1 Score for AU26': individual_f1_scores[11].item(),
                         'Average of F1 Scores': torch.mean(individual_f1_scores).item(),
                         'Overall F1 Score': average_f1_score.item()}

            # Calculate Accuracies
            num_predictions = self.true_positives[0] + self.true_negatives[0] \
                                      + self.false_positives[0] + self.false_negatives[0]
            accuracy_tensor = (self.true_positives + self.true_negatives) / num_predictions

            accuracies = {'Accuracy for AU1': accuracy_tensor[0].item(),
                          'Accuracy for AU2': accuracy_tensor[1].item(),
                          'Accuracy for AU4': accuracy_tensor[2].item(),
                          'Accuracy for AU5': accuracy_tensor[3].item(),
                          'Accuracy for AU6': accuracy_tensor[4].item(),
                          'Accuracy for AU9': accuracy_tensor[5].item(),
                          'Accuracy for AU12': accuracy_tensor[6].item(),
                          'Accuracy for AU15': accuracy_tensor[7].item(),
                          'Accuracy for AU17': accuracy_tensor[8].item(),
                          'Accuracy for AU20': accuracy_tensor[9].item(),
                          'Accuracy for AU25': accuracy_tensor[10].item(),
                          'Accuracy for AU26': accuracy_tensor[11].item(),
                          'Average Accuracy': torch.mean(accuracy_tensor).item()}

            # Calculate 2AFC Score
            component_2afc_tensor, average_2afc = metric_utils.compute_2AFC(self.au_ground_truths,
                                                                            self.au_predictions)
            values_2afc = {'2AFC Score for AU1': component_2afc_tensor[0].item(),
                           '2AFC Score for AU2': component_2afc_tensor[1].item(),
                           '2AFC Score for AU4': component_2afc_tensor[2].item(),
                           '2AFC Score for AU5': component_2afc_tensor[3].item(),
                           '2AFC Score for AU6': component_2afc_tensor[4].item(),
                           '2AFC Score for AU9': component_2afc_tensor[5].item(),
                           '2AFC Score for AU12': component_2afc_tensor[6].item(),
                           '2AFC Score for AU15': component_2afc_tensor[7].item(),
                           '2AFC Score for AU17': component_2afc_tensor[8].item(),
                           '2AFC Score for AU20': component_2afc_tensor[9].item(),
                           '2AFC Score for AU25': component_2afc_tensor[10].item(),
                           '2AFC Score for AU26': component_2afc_tensor[11].item(),
                           'Overall 2AFC Score': average_2afc.item()}

            scalar_dict = OrderedDict(
                [('Similarity Measure between z1 and z2', self.z1_z2_similarity_value.item()),
                 ('Variability in z2', self.z2_z2_variability_value.item()),
                 ('F1 Scores', f1_scores),
                 ('Accuracies', accuracies),
                 ('2AFC Scores', values_2afc)])

        else: # when testing, we want to calculate the F1 scores across several batches
            scalar_dict = OrderedDict(
                [('Similarity measure between z1 and z2', self.z1_z2_similarity_value.item()),
                 ('Variability in z2', self.z2_z2_variability_value.item()),
                 ('true_pos', self.true_positives),
                 ('true_neg', self.true_negatives),
                 ('false_pos', self.false_positives),
                 ('false_neg', self.false_negatives),
                 ('au_predictions', self.au_predictions),
                 ('au_ground_truths', self.au_ground_truths)])

        return scalar_dict
