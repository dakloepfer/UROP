'''Script to parse Action Unit Activations (Intensities) and save them in a pickled dictionary'''

import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np


ARGS = argparse.ArgumentParser()
ARGS.add_argument('--raw_au_path', type=str,
                  help='path to directory containing the raw AUs as provided by DISFA')
ARGS.add_argument('--data_dir', type=str,
                  help='path to directory where the pickled dictionary will be saved')
ARGS.add_argument('--name', type=str, default='aus_disfa',
                  help='name of pickled dictionary file without extension')
ARGS.add_argument('--convert_to_occurrence', action='store_true', default=False,
                  help='boolean whether the AU intensities are converted to occurrences, with active meaning intensity > 2')
OPT = ARGS.parse_args()

def load_all_aus(directory):
    '''
        loads all action units and returns finished dictionary
    '''
    total_au_dict = {}

    for video_directory in tqdm(os.listdir(directory), desc='Going through videos...'):
        total_au_dict.update(load_single_video_aus(os.path.join(directory, video_directory)))

    return total_au_dict

def load_single_video_aus(vid_directory):
    '''
        returns dictionary for single video, as specified by vid_directory
    '''

    list_of_au_values = []
    file_list = sorted(os.listdir(vid_directory), key=lambda x: (len(x), x))
    for filename in tqdm(file_list, desc='Loading video AUs...'):
        filepath = os.path.join(vid_directory, filename)
        list_of_au_values.append(load_single_video_single_au(filepath))

    au_array = np.transpose(np.array(list_of_au_values))

    video_au_dict = {}
    for frame_ind in tqdm(range(0, au_array.shape[0]), desc='Adding to dictionary...'):

        frame_id = vid_directory.split('/')[-1] + '_' + str(frame_ind)
        video_au_dict[frame_id] = au_array[frame_ind]
    return video_au_dict

def load_single_video_single_au(filepath):
    '''
        returns a list of the action unit activations for a given action unit by frame
    '''
    with open(filepath, 'r') as raw_data:
        if OPT.convert_to_occurrence:
            au_activations = [float(line.split(',')[1].strip()) > 2 for line in raw_data]
        else:
            au_activations = [float(line.split(',')[1].strip()) for line in raw_data]

    return au_activations

def main():
    '''
        takes in the action units as provided by DISFA (saved in raw_au_path) and saves
        them in a pickled dictionary as required by GANimation in data_dir;
        format is {'FrameId': numpy array}
    '''
    au_dict = load_all_aus(OPT.raw_au_path)

    with open(os.path.join(OPT.data_dir, OPT.name + '.pkl'), 'wb') as save_file:
        pickle.dump(au_dict, save_file, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    main()
