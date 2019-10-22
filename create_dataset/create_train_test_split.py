'''
    Script to randomly partition dataset into train/test sets, by naively choosing 
    random frames to go into each set
'''
import os
import argparse
import random
import numpy as np 

ARGUMENTS = argparse.ArgumentParser()
ARGUMENTS.add_argument('--image_dir', type=str,
                       help='path to directory containing all available images')
ARGUMENTS.add_argument('--frac_test', type=float, default=0.3,
                       help='fraction of dataset that is tested on')
ARGUMENTS.add_argument('--dataset_dir', type=str,
                       help='path to dataset dir, where test_ids.csv & train_ids.csv are saved')
OPT = ARGUMENTS.parse_args()


def main():
    '''
        create train_ids.csv and test_ids.csv files by randomly partitioning
        the dataset according to frac_train
    '''

    id_list = [img_id.split('.')[0] + '\n' for img_id in os.listdir(OPT.image_dir)]
    n_obj = len(id_list)
    id_list = np.array(id_list)

    test_indices = random.sample(range(n_obj), int(np.ceil(OPT.frac_test * n_obj)))
    train_indices = [ind for ind in range(n_obj) if ind not in test_indices]

    train_file_path = os.path.join(OPT.dataset_dir, 'train_ids.csv')
    with open(train_file_path, 'w') as train_file:
        train_file.writelines(id_list[train_indices].tolist())

    test_file_path = os.path.join(OPT.dataset_dir, 'test_ids.csv')
    with open(test_file_path, 'w') as test_file:
        test_file.writelines(id_list[test_indices].tolist())


if __name__ == '__main__':
    main()
