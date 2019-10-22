'''File to test how well z1 and z2 have disentangled Action Unit from other features'''
import time
import os
from collections import OrderedDict
import torch
import torchvision

from train import Train
from networks.test_disentangling_model import TestDisentanglingModel
from networks.network_component import NetworkComponent

import utils.visualisation_utils as visualisation_utils

class TestDisentangling(Train):
    '''Wrapper Class for code to test disentangling of features'''

    def __init__(self):
        super(TestDisentangling, self).__init__()

        save_path = os.path.join(self.args.save_dir, self.args.name)
        self.model = TestDisentanglingModel(self.cfg, self.args, is_train=True, save_path=save_path)

    def log_train_results(self, _, __, gradient_flow_plots, n_epoch):
        '''
            Logs training errors, test errors, and gradient flow plots to tensorboard
        '''
        # Log losses
        self.tb_visualiser.plot_scalars(self.model.get_current_errors(),
                                        self.total_steps, is_train=True)
        self.log_test_errors(n_epoch)

        self.tb_visualiser.plot_figures(gradient_flow_plots,
                                        self.total_steps, is_train=True)

    def log_test_errors(self, n_epoch):
        '''
            Calculates errors on test dataset and logs them using TBVisualiser
        '''
        test_start_time = time.time()

        # Set maximum number of batches to test
        max_test_iters = min(self.cfg['General'][0]['max_test_iters'],
                             len(self.dataloader_test) // self.batch_size)
        if max_test_iters == 0:
            max_test_iters = 1

        # Set model to evaluation mode
        self.model.is_train = False
        self.model.eval()

        # Evaluate max_test_iters batches
        test_errors = OrderedDict()
        for i_test_batch, test_batch in enumerate(self.dataloader_test):

            if i_test_batch == max_test_iters:
                break

            self.model.forward(test_batch)
            errors = self.model.get_current_errors()

            # Save errors from current batch
            for label, error in errors.items():
                if label in test_errors:
                    test_errors[label] += error
                else:
                    test_errors[label] = error

        # Normalise errors
        for label in test_errors.keys():
            test_errors[label] /= max_test_iters

        # Log to tensorboard
        test_duration = time.time() - test_start_time
        self.tb_visualiser.plot_scalars(test_errors, self.total_steps, is_train=False)
        visualisation_utils.print_current_test_errors(n_epoch, test_duration, test_errors)

        # Set model back to training mode
        self.model.is_train = True
        self.model.train()


    def create_images_from_shuffled_encodings(self, n_batches):
        '''
            This loads the encoder and decoder, and creates n_batch * batch_size images
            from shuffled encodings.
            These images are saved in save_dir
        '''
        print('Creating images from shuffled encodings:')
        encoder = NetworkComponent(self.cfg['Encoder'], self.args, name='Encoder')\
                  .to(self.args.device)
        encoder.load()

        decoder = NetworkComponent(self.cfg['Decoder'], self.args, name='Decoder')\
                  .to(self.args.device)
        decoder.load()

        for i_batch, batch in enumerate(self.dataloader_test):
            if i_batch >= n_batches:
                break

            print('Starting Batch %d...' %i_batch)

            # Read in image
            image = batch[0].to(device=self.args.device)

            real_encoding = encoder.forward(image)

            # Read out z1 and z2
            z_1 = real_encoding[:, :self.cfg['AUPredictor'][0]['channels']]
            z_2 = real_encoding[:, self.cfg['AUPredictor'][0]['channels']:]

            # Move z2 elements by one index, because the dataloader shuffles
            # the batches, this corresponds to shuffling the (z1, z2) pairs
            inds = [ind for ind in range(1, z_2.size()[0])] + [0]
            z_2 = z_2[inds]

            # Merge z1 and z2 back together
            shuffled_encoding = torch.cat((z_1, z_2), dim=1)

            real_reconstructed_imgs = decoder.forward(real_encoding)
            shuffled_reconstructed_imgs = decoder.forward(shuffled_encoding)

            print('Logging images for batch %d' %i_batch)

            # Log images
            for ind in range(self.batch_size):
                original_image = visualisation_utils.tensor2im(image[ind])
                real_rec_image = visualisation_utils.tensor2im(real_reconstructed_imgs[ind])
                shuff_rec_image = visualisation_utils.tensor2im(shuffled_reconstructed_imgs[ind])

                torchvision.utils.save_image([original_image, real_rec_image, shuff_rec_image],
                                             os.path.join(self.args.log_dir, self.args.name,
                                                          'images_ind_%d_batch_%d.jpg'
                                                          % (ind, i_batch)))
                #self.tb_visualiser.log_images({'Original Image': original_image,
                #                               'Real Reconstructed Image': real_rec_image,
                #                               'Shuffled Reconstructed Image': shuff_rec_image},
                #                              i_batch * self.batch_size + ind, is_train=False)

            print('Batch %d finished' %i_batch)



if __name__ == '__main__':
    T = TestDisentangling()
    #T.train()
    T.create_images_from_shuffled_encodings(1)
