'''File containing the TBVisualiser Class and other helper functions for visualisations'''

import os
import time
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tensorboardX import SummaryWriter


class TBVisualiser:
    '''
        Class containing all files and information
        needed to create Tensorboard visualisations
    '''
    def __init__(self, log_path):

        if not os.path.isdir(log_path):
            os.makedirs(log_path)

        self.writer = SummaryWriter(log_path)


    def plot_scalars(self, scalars, iteration, is_train):
        '''
            Parameters:
                scalars: dictionary {label: scalar} containing the scalars
                        to be saved.
                iteration: iteration number under which to save the scalar
                is_train: Boolean deciding whether the scalar is referring to
                        a training or testing run
        '''
        for label, scalar in scalars.items():

            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)

            if isinstance(scalar, float) or isinstance(scalar, int):
                self.writer.add_scalar(sum_name, scalar, iteration)
            else: # scalar is actually a dictionary of scalars
                self.writer.add_scalars(sum_name, scalar, iteration)


    def plot_figures(self, figures, iteration, is_train):
        '''
            Parameters:
                figures: dictionary {label: figure} containing matplotlib plot_figures
                        to be logged
                iteration: iteration number under which to save the figure
                is_train: Boolean deciding whether the scalar is referring to a
                        training or testing run
        '''

        for label, figure in figures.items():

            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            self.writer.add_figure(sum_name, figure, iteration)

    def log_images(self, images, iteration, is_train):
        '''
            Parameters:
                images: dictionary {label: image} containing images to be logged
                iteration: iteration number under which to save the image
                is_train: Boolean deciding whether the scalar is referring to a
                        training or testing run
        '''

        for label, image in images.items():

            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            self.writer.add_image(sum_name, image, iteration)

def print_current_train_errors(n_epoch, i_train_batch, iters_per_epoch,
                               time_per_sample, errors):
    '''
        Prints the current errors for the training set.
        Parameters:
            n_epoch: current epoch number
            i_train_batch: batch number in current epoch
            iters_per_epoch: number of batches per epoch
            time_per_sample: Time taken for training per sample
            errors: dictionary containing error name and error value
                    as (key, value) pairs
    '''

    current_time = time.strftime('[%d/%m/%Y %H:%M:%S]')
    message = '\n%s: Current Training Set Errors \n\
    (epoch: %d, iterations in epoch: %d / %d, time per sample: %.3f seconds)\n' \
                % (current_time, n_epoch, i_train_batch, iters_per_epoch, time_per_sample)

    for label, error in errors.items():
        message += '%s: %.3f\n' %(label, error)

    print(message)

def print_current_test_errors(n_epoch, test_duration, errors):
    '''
        Prints the current errors for the test set.
        Parameters:
            n_epoch: current epoch number
            test_duration: Time taken for testing
            errors: dictionary containing error name and error value
                    as (key, value) pairs
    '''

    current_time = time.strftime('[%d/%m/%Y %H:%M:%S]')
    message = '\n%s: Current Test Set Errors \n\
    (epoch: %d, time taken for testing: %.3f\n' \
                % (current_time, n_epoch, test_duration)

    for label, error in errors.items():
        message += '%s: %.3f\n' %(label, error)

    print(message)

def tensor2im(image):
    '''
        Turn tensor describing image to unnormalised image that can be used by tensorboardX.
        Parameters:
            image: The image tensor
            unnormalise: Boolean deciding if the image needs to be unnormalised
    '''
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    for img_el, mean_el, std_el in zip(image, mean, std):
        img_el.mul_(std_el).add_(mean_el)

    return image

def plot_gradient_flow(named_parameters, model_name):
    '''
        Takes in a list of named_parameters and returns a matplotlib
        bar plot of the average and maximum gradients through the layers.

        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        plot_grad_flow(self.model.named_parameters()) to visualize the gradient flow.

        Code adapted from RoshanRane on
        https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''
    avg_grads = []
    max_grads = []
    layer_names = []

    for name, params in named_parameters:

        if(params.requires_grad) and ('bias' not in name):

            layer_names.append(name)
            avg_grads.append(params.grad.abs().mean())
            max_grads.append(params.grad.abs().max())

    if not layer_names: # no weights provided, return empty plot
        return plt.subplots()

    grad_flow_fig, grad_flow_ax = plt.subplots()

    grad_flow_ax.bar(np.arange(1, len(max_grads) + 1), max_grads, lw=1, color='cyan')
    grad_flow_ax.bar(np.arange(1, len(max_grads) + 1), avg_grads, lw=1, color='blue')

    grad_flow_ax.hlines(0, 0, len(avg_grads)+1, lw=2, color='black')
    grad_flow_ax.set_xticks(range(1, len(avg_grads) + 1, 1))
    grad_flow_ax.set_xticklabels(layer_names, rotation='vertical')
    #grad_flow_ax.set_xlim(left=0, right=len(avg_grads) + 1)
    top_lim = max(avg_grads).cpu() * 1.5
    grad_flow_ax.set_ylim(bottom=0, top=top_lim) # zoom in on the lower gradient regions

    grad_flow_ax.set_xlabel('Layers')
    grad_flow_ax.set_ylabel('Gradient')
    grad_flow_ax.set_title('Gradient Flow in %s' %model_name)
    grad_flow_ax.grid(True)
    grad_flow_ax.legend([Line2D([0], [0], color='cyan', lw=4),
                         Line2D([0], [0], color='blue', lw=4),
                         Line2D([0], [0], color='black', lw=4)],
                        ['max gradient', 'mean gradient', 'zero gradient'])

    return grad_flow_fig, grad_flow_ax
