'''Script to create AU occurrence dictionary for the BP4D dataset as provided by Tian'''

import os
import pickle
import argparse
import numpy as np

ARGS = argparse.ArgumentParser()
ARGS.add_argument('--raw_au_path', type=str,
                  help='path to .txt file containing the raw AUs as provided by BP4D')
ARGS.add_argument('--data_dir', type=str,
                  help='path to directory where the pickled dictionary will be saved')
ARGS.add_argument('--name', type=str, default='aus_bp4d_occurrence',
                  help='name of pickled dictionary file without extension')
OPT = ARGS.parse_args()

def main():
    '''
        Takes in the action units as provided by BP4D (saved in raw_au_path) and saves
        them in a pickled dictionary as required by GANimation in data_dir;
        format is {'FrameId': numpy array}
    '''
    with open(OPT.raw_au_path, 'r') as au_file:
        lines = [line.strip().split(' ') for line in au_file]

    frame_ids = [line[0].split('.')[0].replace('/', '_') for line in lines]
    occurrences = [np.array([int(au_occurrence) for au_occurrence in line[1:]]) for line in lines]

    au_dict = {}
    for index, frame_id in enumerate(frame_ids):
        au_dict[frame_id] = occurrences[index]

    with open(os.path.join(OPT.data_dir, OPT.name + '.pkl'), 'wb') as save_file:
        pickle.dump(au_dict, save_file, pickle.HIGHEST_PROTOCOL)

    print('Finished without error.')

if __name__ == '__main__':
    main()
