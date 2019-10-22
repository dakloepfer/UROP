'''Script to investigate the imbalance in Action Unit Occurrence in a dataset'''

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

ARGS = argparse.ArgumentParser()
ARGS.add_argument('--au_filepath', type=str,
                  help='path to pickled AU dictionary')
ARGS.add_argument('--occurrence_cutoff', type=float,
                  help='Cutoff (scale 0-5) above which AU intensities count as active')
OPT = ARGS.parse_args()

def count_occurrences(au_dict):
    '''
        Parameters:
            au_dict: dictionary of the format {'FrameId': numpy array (length n)}
                    containing the AU intensities
        Returns:
            a numpy array of length n+1 containing the numbers
            of frames in which the given Action Unit occurs, with the last
            element being the number of frames in which no Action Unit is
            active
    '''
    number_neutral_frames = 0
    au_occurrence_numbers = np.zeros_like(next(iter(au_dict.values())))

    for _, intensities in au_dict.items():

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
              %(label, occurrence_numbers[i], 100 * occurrence_numbers[i] / total_frame_number))
    print('\n')

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

def count_co_occurrences(au_dict):
    '''
        Parameters:
            au_dict: dictionary of the format {'FrameId': numpy array (length n)}
                    containing the AU intensities
        Returns:
            au_occurrence_numbers: a numpy array of dimension n x n that contains
                in row i the numbers of active frames for the action units among the frames
                where action unit i was active.
    '''
    number_of_aus = len(next(iter(au_dict.values())))
    au_occurrence_numbers = np.zeros((number_of_aus, number_of_aus))

    for au_ind in range(number_of_aus):

        for _, intensities in au_dict.items():

            occurrences = intensities > OPT.occurrence_cutoff

            if occurrences[au_ind]:
                au_occurrence_numbers[au_ind] += occurrences

    return au_occurrence_numbers

def visualise_co_occurrences(co_occurences, total_occurrences, labels=None):
    '''
        Parameters:
            co_occurences: n x n numpy array containing in row i the numbers of
                        frames in which a given AU is active among the frames in which
                        AU at index i is active
            total_occurrences: total number of frames in which AUs are active, as returned
                        by count_occurrences()
            labels: labels for the elements of co_occurences

        Behaviour:
            Prints and plots occurrence_numbers as a histogram
    '''
    if labels is None:
        labels = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15',
                  'AU17', 'AU20', 'AU25', 'AU26']

    for row_ind, row_label in enumerate(labels):

        # Print results
        for i, au_label in enumerate(labels):
            print('Number of active frames for %s given that %s is active: %d in total, \
                   %f%% of all active frames for %s'
                  % (au_label, row_label, co_occurences[row_ind, i],
                     100 * co_occurences[row_ind, i] / total_occurrences[i], au_label))
        print('\n')

        # Make histogram
        _, axs = plt.subplots()
        axs.bar(np.arange(1, len(labels) + 1), co_occurences[row_ind, :])
        axs.set_xticks(range(1, len(labels) + 1, 1))
        axs.set_xticklabels(labels)
        axs.set_ylabel('Number of Active Frames')
        axs.grid(True)
        axs.set_title('Number of Active Frames Given That %s is Active' % row_label)

        plt.savefig('disfa_dataset/active_frames_if_%s_active.png' %row_label)

    plt.show()


def main():
    '''Main wrapper script'''

    with open(OPT.au_filepath, 'rb') as au_file:
        au_dict = pickle.load(au_file)

    occurrence_numbers = count_occurrences(au_dict)

    visualise_occurrence_numbers(occurrence_numbers, len(au_dict))

    co_occurrences = count_co_occurrences(au_dict)

    visualise_co_occurrences(co_occurrences, occurrence_numbers)

if __name__ == '__main__':
    main()
