'''
    Scripts to create a train_ids.csv or test_ids.csv file out of
    frames from a given list of subjects according to the
    MOMU algorithm.
    File includes code to visualise the balance in AU occurrences
    of the resulting lists.
'''

import argparse
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

ARGS = argparse.ArgumentParser()
ARGS.add_argument('--au_filepath', type=str,
                  help='path to pickled AU dictionary')
ARGS.add_argument('--occurrence_cutoff', type=float,
                  help='Cutoff above which AU intensities count as active')
ARGS.add_argument('--img_dir', type=str,
                  help='path to image directory, used to get a list of all frame_ids')
ARGS.add_argument('--allowed_subjects_file', type=str,
                  help='File containing IDs of subjects allowed to use, one per line')
ARGS.add_argument('--create_dataset', action='store_true', default=False,
                  help='Decides if the user wants to experiment with MOMU or create training set')
ARGS.add_argument('--save_dir', type=str, help='Directory in which ID file will be saved')
ARGS.add_argument('--file_name', type=str, help='Name of File containing frame IDs')
OPT = ARGS.parse_args()

def momu(dataset_list, batch_size, sampling_size, number_of_batches=1):
    '''
        Implement the MOMU Algorithm from Chu et.al. (2019)
        "Learning facial action units with spatiotemporal cues and multi-label sampling."

        Parameters:
            dataset_list: a list of unique frame ids that are available in the dataset
            batch_size: integer, size of one batch
            sampling_size: integer, number of elements that are taken from dataset_list
                    in one go, must divide batch_size
            number_of_batches: integer, number of batches to be produced

        Returns:
            List of batches, the batches are lists of frame indices in that batch.
    '''

    assert batch_size % sampling_size == 0, 'sampling_size must divide batch_size!'

    n_samplings_per_batch = int(batch_size / sampling_size)

    with open(OPT.au_filepath, 'rb') as au_file:
        au_dict = pickle.load(au_file)

    n_classes = len(next(iter(au_dict.values()))) + 1 # number of AUs + 1 for neutral expression

    # Create list that at position i contains a set containing frame ids with the
    # the action unit class i active
    dataset = [set() for i in range(n_classes)]
    complete_dataset = [[] for i in range(n_classes)] # needed to restore elements to dataset

    for frame_id in dataset_list:

        active_classes = au_dict[frame_id] > OPT.occurrence_cutoff

        if sum(active_classes) == 0: # neutral frame
            dataset[-1].add(frame_id)
            complete_dataset[-1].append(frame_id)
        else:
            for class_index, _ in enumerate(active_classes):
                if active_classes[class_index]:
                    dataset[class_index].add(frame_id)
                    complete_dataset[class_index].append(frame_id)

    batches = []
    for _ in range(number_of_batches):

        current_class_index = dataset.index(min(dataset, key=len))
        current_batch = []
        current_batch_class_occurrences = np.zeros(n_classes)

        for _ in range(n_samplings_per_batch):

            # restore elements to dataset if necessary
            if len(dataset[current_class_index]) < sampling_size:
                # restore elements all classes
                for frame_id in complete_dataset[current_class_index]:
                    frame_occurrences = au_dict[frame_id] > OPT.occurrence_cutoff

                    if np.sum(frame_occurrences) == 0:
                        frame_occurrences = np.append(frame_occurrences, 1)
                    else:
                        frame_occurrences = np.append(frame_occurrences, 0)

                    for class_ind, class_set in enumerate(dataset):
                        if frame_occurrences[class_ind]:
                            class_set.add(frame_id)

            # Sample sampling_size elements from current_class_index
            sampled_frames = random.sample(dataset[current_class_index], sampling_size)

            # Remove sampled_frames from dataset, update occurrences in batch
            for frame_id in sampled_frames:
                frame_occurrences = au_dict[frame_id] > OPT.occurrence_cutoff

                if np.sum(frame_occurrences) == 0:
                    frame_occurrences = np.append(frame_occurrences, 1)
                else:
                    frame_occurrences = np.append(frame_occurrences, 0)
                current_batch_class_occurrences += frame_occurrences

                for class_ind, class_set in enumerate(dataset):
                    if frame_occurrences[class_ind]:
                        class_set.remove(frame_id)

            # Add sampled_frames to batch
            current_batch += sampled_frames

            # update current_class_index
            current_class_index = np.where(
                current_batch_class_occurrences \
                == np.amin(current_batch_class_occurrences))[0][0]

        batches.append(current_batch)

    return batches

def count_occurrences(frame_id_list):
    '''
        Parameters:
            frame_id_list: list of frame ids across which the AU occurrences
                    are counted.
        Returns:
            a numpy array of length n+1 containing the numbers
            of frames in which the given Action Unit occurs, with the last
            element being the number of frames in which no Action Unit is
            active
    '''

    with open(OPT.au_filepath, 'rb') as au_file:
        au_dict = pickle.load(au_file)

    number_neutral_frames = 0
    au_occurrence_numbers = np.zeros_like(next(iter(au_dict.values())))

    for frame_id in frame_id_list:

        intensities = au_dict[frame_id]

        occurrences = intensities > OPT.occurrence_cutoff

        if np.sum(occurrences) == 0:
            number_neutral_frames += 1

        au_occurrence_numbers += occurrences

    return np.append(au_occurrence_numbers, number_neutral_frames)

def visualise_occurrence_numbers(occurrence_numbers, total_frame_number, labels=None):
    '''
        Parameters:
            occurrence_numbers: numpy array containing numbers of
                        frames in which a given AU is active
            total_frame_number: total number of frames, used to calculate percentages
            labels: labels for the elements of occurrence_numbers

        Behaviour:
            Prints and plots occurrence_numbers as a histogram
    '''

    if labels is None:
        labels = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15',
                  'AU17', 'AU20', 'AU25', 'AU26', 'Neutral']

    # Print results
    for i, label in enumerate(labels):
        print('Number of active frames for %s: %d in total, %f%% of all frames'
              %(label, occurrence_numbers[i], 100 * occurrence_numbers[i] / total_frame_number),
              file=OUT_FILE)
    print('\n', file=OUT_FILE)

    # Make histogram
    _, all_ax = plt.subplots()
    all_ax.bar(np.arange(1, len(labels) + 1), occurrence_numbers)
    all_ax.set_xticks(range(1, len(labels) + 1, 1))
    all_ax.set_xticklabels(labels)
    all_ax.set_ylabel('Number of Active Frames')
    all_ax.grid(True)
    all_ax.set_title('Number of Active Frames Including Neutral Frames')

    _, au_ax = plt.subplots()
    au_ax.bar(np.arange(1, len(labels)), occurrence_numbers[:-1])
    au_ax.set_xticks(range(1, len(labels), 1))
    au_ax.set_xticklabels(labels[:-1])
    au_ax.set_ylabel('Number of Active Frames')
    au_ax.grid(True)
    au_ax.set_title('Number of Active Frames for AUs only')

    plt.show()

def experiment():
    '''
        Main Script for experimenting with the MOMU Algorithm.
        User needs to define the source of the complete list of frame IDs,
        and may change the parameters to their liking.
    '''
    with open(OPT.allowed_subjects_file, 'r') as allowed_subjects_file:
        allowed_subjects = [line.strip() for line in allowed_subjects_file]

    frame_list = [filename.split('.')[0] for filename in os.listdir(OPT.img_dir) \
                   if filename.split('_')[0] in allowed_subjects]

    #with open('disfa_dataset/train_ids.csv', 'r') as train_file:
    #    train_ids = [line.strip() for line in train_file \
    #                 if line.split('_')[0] in allowed_subjects]
    #
    #with open('disfa_dataset/test_ids.csv', 'r') as test_file:
    #    test_ids = [line.strip() for line in test_file \
    #                if line.split('_')[0] in allowed_subjects]
    #
    #frame_list = train_ids + test_ids
    batch_size = 48480 #96928
    sampling_size = 32
    number_of_batches = 1
    batches = momu(frame_list, batch_size, sampling_size, number_of_batches)

    print('\n\nNew Experiment:\nBatch Size: %d\nSampling Size: %d\nNumber of Batches: %d\n'
          %(batch_size, sampling_size, number_of_batches), file=OUT_FILE)

    n_duplicates = batch_size * number_of_batches \
                 - len(set([frame for batch in batches for frame in batch]))

    print('\nNumber of duplicates: %d, %f%% of all elements in dataset\n'
          %(n_duplicates, 100 * n_duplicates / (batch_size*number_of_batches)), file=OUT_FILE)

    for batch_ind, batch in enumerate(batches):
        print('Batch number:', batch_ind, file=OUT_FILE)

        batch_occurence_numbers = count_occurrences(batch)
        visualise_occurrence_numbers(batch_occurence_numbers, len(batch))

def create_dataset():
    '''
        Main script to actually create a dataset and write the
        batches to a datafile.
        User may need to adjust parameters to their own source data.
    '''
    with open(OPT.allowed_subjects_file, 'r') as allowed_subjects_file:
        allowed_subjects = [line.strip() for line in allowed_subjects_file]

    frame_list = [filename.split('.')[0] for filename in os.listdir(OPT.img_dir) \
                   if filename.split('_')[0] in allowed_subjects]

    #with open('disfa_dataset/train_ids.csv', 'r') as train_file:
    #    train_ids = [line.strip() for line in train_file \
    #                 if line.split('_')[0] in allowed_subjects]
    #
    #with open('disfa_dataset/test_ids.csv', 'r') as test_file:
    #    test_ids = [line.strip() for line in test_file \
    #                if line.split('_')[0] in allowed_subjects]
    #
    #frame_list = train_ids + test_ids
    batch_size = 48480
    sampling_size = 32
    number_of_batches = 1
    batches = momu(frame_list, batch_size, sampling_size, number_of_batches)

    with open(os.path.join(OPT.save_dir, OPT.file_name), 'w+') as output_file:

        for batch in batches:
            output_file.writelines([frame + '\n' for frame in batch])


if __name__ == '__main__':
    if not OPT.create_dataset:
        OUT_FILE = open('create_dataset/momu_bp4d_experimentation.txt', 'a')
        experiment()
        OUT_FILE.close()

    else:
        create_dataset()
