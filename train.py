''' Main file. Contains code for training and periodic validation, similar to GANimation.'''

import time
import os
import sys
import argparse
import math
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader

from networks.model import Model
from data.dataset import ImageWithAUDataset
from utils.cfg_utils import parse_cfg
import utils.visualisation_utils as visualisation_utils
import utils.metric_utils as metric_utils
from utils.visualisation_utils import TBVisualiser

def parse_arguments():
    '''
        Parses command line arguments, returns them as args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, help='Path to .cfg file for network')
    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    parser.add_argument('--train_ids_file', type=str, default='train_ids.csv',
                        help='File containing image ids of training set, path relative to data_dir')
    parser.add_argument('--test_ids_file', type=str, default='test_ids.csv',
                        help='File containing image ids of testing set, path relative to data_dir')
    parser.add_argument('--aus_file', type=str, default='aus_dict.pkl',
                        help='File containing AU activations, path relative to data_dir')
    parser.add_argument('--img_dir', type=str, default='imgs',
                        help='Path from data_dir to image folder')
    parser.add_argument('--name', type=str, default='experiment_1',
                        help='Name of the experiment. Decides where results are stored')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory where all visualisations are stored under experiment name')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory where weights & optimizers are saved under experiment name')
    parser.add_argument('--load_epoch', type=int, default=-1,
                        help='Epoch to load. Set to -1 to use latest cached model')
    parser.add_argument('--load_dir', type=str, help='Directory containing saved weights')
    parser.add_argument('--print_period_s', type=int, default=360,
                        help='Maximum period in sec between showing training results on console')
    parser.add_argument('--log_period_s', type=int, default=300,
                        help='Maximum period in sec between logging train results in tensorboard')
    parser.add_argument('--max_training_time_s', type=float, default=-1.0,
                        help='Maximum time the whole program will run in seconds \
                              - use to stop before being cancelled')


    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args



class Train:
    '''Train class; wrapper for all commands'''
    def __init__(self):
        self.training_start_time = time.time()

        self.args = parse_arguments()
        self.cfg = parse_cfg(self.args.cfg_file)
        self.set_and_check_load_epoch()

        if self.args.max_training_time_s <= 0:
            self.max_training_time = math.inf
        else:
            self.max_training_time = self.args.max_training_time_s

        regress_au_intensities = self.cfg['General'][1]['regress_au_intensities']

        self.dataset_train = ImageWithAUDataset(self.args, self.cfg['General'][1], is_train=True,
                                                regress_au_intensities=regress_au_intensities)
        self.dataset_test = ImageWithAUDataset(self.args, self.cfg['General'][1], is_train=False,
                                               regress_au_intensities=regress_au_intensities)

        self.batch_size = self.cfg['General'][0]['batch_size']
        self.max_epochs = self.cfg['General'][0]['max_epochs']

        self.dataloader_train = DataLoader(self.dataset_train,
                                           batch_size=self.batch_size,
                                           shuffle=True, drop_last=True)
        self.dataloader_test = DataLoader(self.dataset_test,
                                          batch_size=self.batch_size,
                                          shuffle=True, drop_last=True)

        save_path = os.path.join(self.args.save_dir, self.args.name)
        self.model = Model(self.cfg, self.args, is_train=True, save_path=save_path)

        self.tb_visualiser = TBVisualiser(os.path.join(self.args.log_dir, self.args.name))

        self.total_steps = self.args.load_epoch * len(self.dataloader_train) * self.batch_size
        self.iters_per_epoch = len(self.dataset_train) / self.batch_size

        self.last_log_time = None
        self.last_print_time = time.time()

    def train(self):
        '''
            Script that controls training schedule
        '''

        for n_epoch in range(self.args.load_epoch + 1, self.max_epochs + 1):
            epoch_start_time = time.time()

            print('\nStarting epoch %d / %d...\n' %(n_epoch, self.max_epochs))

            if self.cfg['General'][1]['balance_dataset_with_momu']:
                self.dataset_train.run_momu_for_new_epoch()

            # Train one epoch
            self.train_epoch(n_epoch)

            # Save model
            print('saving the model at the end of epoch %d, iters %d' % (n_epoch, self.total_steps))
            self.model.save(n_epoch)

            # Print epoch info
            epoch_duration = time.time() - epoch_start_time
            print('\nEnd of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)\n' %
                  (n_epoch, self.max_epochs, epoch_duration,
                   epoch_duration / 60, epoch_duration / 3600))

            # Update learning rate
            self.model.update_learning_rate(n_epoch)


    def train_epoch(self, n_epoch):
        '''
            This script trains the network for a single epoch
            Parameters:
                n_epoch: number of current epoch
        '''

        epoch_iter = 0
        self.model.is_train = True

        for i_train_batch, train_batch in enumerate(self.dataloader_train):
            iter_start_time = time.time()

            if iter_start_time - self.training_start_time >= self.max_training_time:
                sys.exit()

            # Set logging & printing flags
            log_results = self.last_log_time is None or \
                          time.time() - self.last_log_time > self.args.log_period_s or \
                          i_train_batch == len(self.dataloader_train) - 1
                          # log results at end of epoch

            print_results = log_results or \
                            time.time() - self.last_print_time > self.args.print_period_s

            # Train model
            reconstr_image, _, gradient_flow_plots = \
                    self.model.optimize_parameters(i_train_batch, train_batch, log_results)

            # Update epoch info
            self.total_steps += self.batch_size
            epoch_iter += self.batch_size

            # Print results to console
            if print_results:
                errors = self.model.get_current_errors()
                time_per_sample = (time.time() - iter_start_time) / self.batch_size
                visualisation_utils.print_current_train_errors(n_epoch, i_train_batch,
                                                               self.iters_per_epoch,
                                                               time_per_sample, errors)
                self.last_print_time = time.time()

            # Log results
            if log_results:
                self.log_train_results(train_batch, reconstr_image, gradient_flow_plots, n_epoch)
                self.last_log_time = time.time()

    def log_train_results(self, train_batch, reconstr_image, gradient_flow_plots, n_epoch):
        '''
            Logs training errors, test errors, original image and reconstructed image,
            and gradient flow plots to tensorboard
        '''
        # Log losses
        self.tb_visualiser.plot_scalars(self.model.get_current_errors(),
                                        self.total_steps, is_train=True)
        # Log other metrics
        self.tb_visualiser.plot_scalars(self.model.get_current_scalars(),
                                        self.total_steps, is_train=True)

        # Log images
        original_image = visualisation_utils.tensor2im(train_batch[0][0])
        reconstr_image = visualisation_utils.tensor2im(reconstr_image[0].detach())

        self.tb_visualiser.log_images({'Original Image': original_image,
                                       'Reconstructed Image': reconstr_image},
                                      self.total_steps, is_train=True)

        self.tb_visualiser.plot_figures(gradient_flow_plots,
                                        self.total_steps, is_train=True)

        self.log_test_results(n_epoch)

    def log_test_results(self, n_epoch):
        '''
            Calculates errors on test dataset and logs them using TBVisualiser
        '''
        test_start_time = time.time()

        # Set maximum number of batches to test
        max_test_iters = min(self.cfg['General'][0]['max_test_iters'],
                             len(self.dataloader_test))

        if max_test_iters == 0:
            max_test_iters = 1

        # Set model to evaluation mode
        self.model.is_train = False
        self.model.eval()

        # Evaluate max_test_iters batches
        test_errors = OrderedDict()
        test_scalars = OrderedDict()
        for i_test_batch, test_batch in enumerate(self.dataloader_test):

            if i_test_batch == max_test_iters:
                break

            reconstr_image, _ = self.model.forward(test_batch)
            errors = self.model.get_current_errors()

            # Save errors from current batch
            for label, error in errors.items():
                if label in test_errors:
                    test_errors[label] += error
                else:
                    test_errors[label] = error

            scalars = self.model.get_current_scalars()

            for label, scalar in scalars.items():
                if label in test_scalars:
                    if not label in ['au_predictions', 'au_ground_truths']:
                        test_scalars[label] += scalar
                    else:
                        test_scalars[label] = torch.cat((test_scalars[label], scalar), dim=0)
                else:
                    test_scalars[label] = scalar

            # Log first image and reconstructed image
            if i_test_batch == 0:
                original_image = visualisation_utils.tensor2im(test_batch[0][0])
                reconstr_image = visualisation_utils.tensor2im(reconstr_image[0])

                self.tb_visualiser.log_images({'Original Image': original_image,
                                               'Reconstructed Image': reconstr_image},
                                              self.total_steps, is_train=False)
        # Normalise errors
        for label in test_errors.keys():
            test_errors[label] /= max_test_iters

        # Log errors to tensorboard
        test_duration = time.time() - test_start_time
        self.tb_visualiser.plot_scalars(test_errors, self.total_steps, is_train=False)
        visualisation_utils.print_current_test_errors(n_epoch, test_duration, test_errors)

        # Normalise scalars
        for label in test_scalars.keys():
            if not label in ['au_predictions', 'au_ground_truths']:
                test_scalars[label] /= max_test_iters

        # Calculate F1 scores
        f1_denominator = 2 * test_scalars['true_pos'] + test_scalars['false_neg'] \
                       + test_scalars['false_pos']
        f1_denominator += torch.finfo(torch.float).tiny
        individual_f1_scores = 2 * test_scalars['true_pos'] / f1_denominator

        all_tp = torch.sum(test_scalars['true_pos'])
        all_fn = torch.sum(test_scalars['false_neg'])
        all_fp = torch.sum(test_scalars['false_pos']) + torch.finfo(torch.float).tiny

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
        num_predictions = test_scalars['true_pos'] + test_scalars['true_neg'] \
                            + test_scalars['false_pos'] + test_scalars['false_neg']

        accuracy_tensor = (test_scalars['true_pos'] + test_scalars['true_neg']) / num_predictions

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

        # Calculate 2AFC Scores
        component_2afc_tensor, average_2afc = metric_utils.compute_2AFC(
                        test_scalars['au_ground_truths'], test_scalars['au_predictions'])

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

        del test_scalars['true_pos']
        del test_scalars['true_neg']
        del test_scalars['false_pos']
        del test_scalars['false_neg']
        del test_scalars['au_ground_truths']
        del test_scalars['au_predictions']

        test_scalars['F1 Scores'] = f1_scores
        test_scalars['Accuracies'] = accuracies
        test_scalars['2AFC Scores'] = values_2afc

        # Log Metrics
        self.tb_visualiser.plot_scalars(test_scalars, self.total_steps, is_train=False)

        # Set model back to training mode
        self.model.is_train = True
        self.model.train()

    def set_and_check_load_epoch(self):
        '''
            Checks what checkpoint files exist if we try to load
            the latest model (args.load_epoch == -1), checks if
            requested model checkpoint exists if args.load_epoch >= 0
        '''

        if os.path.exists(self.args.load_dir):

            if self.args.load_epoch == -1:

                load_epoch = 0
                for file in os.listdir(self.args.load_dir):

                    filename, extension = os.path.splitext(file)

                    if extension == '.pth':
                        load_epoch = max(load_epoch, int(filename.split('_')[-1]))

                self.args.load_epoch = load_epoch

            elif self.args.load_epoch > 0:

                found = False
                for file in os.listdir(self.args.load_dir):

                    filename, extension = os.path.splitext(file)
                    if extension == '.pth':

                        found = int(filename.split('_')[-1]) == self.args.load_epoch
                        if found:
                            break

                assert found, 'Model for epoch %i not found' % self.args.load_epoch

        else:
            assert self.args.load_epoch < 1, \
                'Tried to load model, model directory %s does not exist' % self.args.load_dir
            self.args.load_epoch = 0

if __name__ == '__main__':
    Train().train()
