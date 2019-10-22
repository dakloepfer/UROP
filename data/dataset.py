'''File containing class for custom dataset'''

import os
import pickle
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class ImageWithAUDataset(Dataset):
    '''
        Class for the dataset of images with corresponding
        Action Unit (AU) activations
    '''
    def __init__(self, args, cfg, is_train=True, regress_au_intensities=False):
        '''
            Parameters:
                args: Command line arguments
                is_train: Boolean whether we use the training or testing
                       dataset path. Default is True.
                regress_au_intensities: Boolean whether we regress the actual AU
                        intensities (and we need to divide activations by 5) or
                        whether we only predict AU occurrence
        '''

        self.args = args
        self.cfg = cfg

        self.is_train = is_train
        self.regress_au_intensities = regress_au_intensities

        # Declare variables
        self.root = None
        self.img_dir = None
        self.ids = None
        self.au_dict = None
        self.epoch_size = None

        self.noise_factor = None

        # Needed when using MOMU for each epoch
        self.complete_dataset = None
        self.dataset = None
        self.sampling_size = None

        self.init_dataset()
        self.create_transform()

    def __len__(self):
        '''
            Returns size of dataset
        '''
        return self.epoch_size

    def __getitem__(self, index):
        '''
            Returns (image, action units, image_id) tuple determined by index
        '''

        # Read image ID
        sample_id = self.ids[index]

        # Read image
        sample_filepath = os.path.join(self.img_dir, sample_id + '.jpg')
        image = cv2.imread(sample_filepath, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read Action Unit Activations, and normalise if necessary
        if self.regress_au_intensities:
            au_activations = self.au_dict[sample_id] / 5.0
        else:
            au_activations = self.au_dict[sample_id]

        # Apply transformations
        image = Image.fromarray(image)

        if self.is_train:

            image = self.augment_data(image)

            self.noise_factor = random.uniform(0.0, 0.015)
            noisy_image = self.apply_noise(image)
            noisy_image = self.transform_to_tensor(noisy_image)

            image = self.transform_to_tensor(image)

            return (noisy_image, image, au_activations, sample_id)

        else:
            image = self.transform_to_tensor(image)

            return (image, au_activations, sample_id)


    def init_dataset(self):
        '''
            Reads in the IDs of the images in the dataset,
            the action unit dictionary,
            and the filepath to the image directory
        '''
        # Read path to image directory
        self.root = self.args.data_dir
        self.img_dir = os.path.join(self.root, self.args.img_dir)

        # Read image IDs
        use_id_filename = self.args.train_ids_file if self.is_train else self.args.test_ids_file
        use_id_filepath = os.path.join(self.root, use_id_filename)
        self.ids = np.loadtxt(use_id_filepath, dtype=np.str, delimiter='\t').tolist()

        # Read Action Unit Activations
        au_filepath = os.path.join(self.root, self.args.aus_file)
        with open(au_filepath, 'rb') as au_file:
            self.au_dict = pickle.load(au_file)

        # Filter IDs to only include the ones we know the action units for
        self.ids = [frame_id for frame_id in self.ids if frame_id in self.au_dict.keys()]

        self.epoch_size = len(self.ids)

        # Initialise things needed for MOMU
        if self.cfg['balance_dataset_with_momu'] and self.is_train:
            self.epoch_size = self.cfg['epoch_size']
            self.sampling_size = self.cfg['sampling_size']

            assert self.epoch_size % self.sampling_size == 0, \
                   'sampling_size must divide epoch_size in cfg file!'

            # number of AUs + 1 for neutral expression
            n_classes = len(next(iter(self.au_dict.values()))) + 1

            # Create list that at position i contains a set containing frame ids with the
            # the action unit class i active
            self.dataset = [set() for i in range(n_classes)]
            # needed to restore elements to dataset
            self.complete_dataset = [[] for i in range(n_classes)]

            occurrence_cutoff = 2 if self.cfg['regress_au_intensities'] else 0.5
            for frame_id in self.ids:

                active_classes = self.au_dict[frame_id] > occurrence_cutoff

                if sum(active_classes) == 0: # neutral frame
                    self.dataset[-1].add(frame_id)
                    self.complete_dataset[-1].append(frame_id)
                else:
                    for class_index, _ in enumerate(active_classes):
                        if active_classes[class_index]:
                            self.dataset[class_index].add(frame_id)
                            self.complete_dataset[class_index].append(frame_id)

    def run_momu_for_new_epoch(self):
        '''
            When called, runs the MOMU algorithm to create a new list of IDs for the new epoch,
            which are saved in self.ids.
            These epochs correspond to the batches in the description of the MOMU algorithm in
            Chu et.al. (2019):
            "Learning facial action units with spatiotemporal cues and multi-label sampling."
        '''
        n_samplings_per_epoch = int(self.epoch_size / self.sampling_size)
        current_class_index = self.dataset.index(min(self.dataset, key=len))
        current_epoch_class_occurrences = np.zeros(len(self.dataset))
        occurrence_cutoff = 2 if self.cfg['regress_au_intensities'] else 0.5

        self.ids = []

        for _ in range(n_samplings_per_epoch):

            # restore elements to dataset if necessary
            if len(self.dataset[current_class_index]) < self.sampling_size:
                # restore elements all classes
                for frame_id in self.complete_dataset[current_class_index]:
                    frame_occurrences = self.au_dict[frame_id] > occurrence_cutoff

                    if np.sum(frame_occurrences) == 0:
                        frame_occurrences = np.append(frame_occurrences, 1)
                    else:
                        frame_occurrences = np.append(frame_occurrences, 0)

                    for class_ind, class_set in enumerate(self.dataset):
                        if frame_occurrences[class_ind]:
                            class_set.add(frame_id)

            # Sample sampling_size elements from current_class_index
            sampled_frames = random.sample(self.dataset[current_class_index], self.sampling_size)

            # Remove sampled_frames from dataset, update occurrences in self.ids
            for frame_id in sampled_frames:
                frame_occurrences = self.au_dict[frame_id] > occurrence_cutoff

                if np.sum(frame_occurrences) == 0:
                    frame_occurrences = np.append(frame_occurrences, 1)
                else:
                    frame_occurrences = np.append(frame_occurrences, 0)
                current_epoch_class_occurrences += frame_occurrences

                for class_ind, class_set in enumerate(self.dataset):
                    if frame_occurrences[class_ind]:
                        class_set.remove(frame_id)

            # Add sampled_frames to self.ids
            self.ids += sampled_frames

            # update current_class_index
            current_class_index = np.where(
                current_epoch_class_occurrences \
                == np.amin(current_epoch_class_occurrences))[0][0]


    def create_transform(self):
        '''
            Defines the transformation applied to the images before being returned.
            This can be extended as required
        '''
        to_tensor = transforms.ToTensor()
        to_img = transforms.ToPILImage()

        self.apply_noise = transforms.Lambda(
            lambda x: to_img(torch.clamp(
                to_tensor(x) + self.noise_factor*torch.randn_like(to_tensor(x)), 0.0, 1.0)))

        change_colour = transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5),
                                               saturation=(0, 1.5), hue=0)

        transform_list = [transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomApply([change_colour], p=0.9)]

        self.augment_data = transforms.Compose(transform_list)

        if not 'mean' in self.cfg.keys():
            mean = [0.5, 0.5, 0.5]
        else:
            mean = [float(val) for val in self.cfg['mean'].split(',')]

        if not 'std' in self.cfg.keys():
            std = [0.5, 0.5, 0.5]
        else:
            std = [float(val) for val in self.cfg['std'].split(',')]

        self.transform_to_tensor = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=mean,
                                                                            std=std)])
