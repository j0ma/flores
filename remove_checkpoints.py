import os
import re

dont_delete_these = {'checkpoint_best.pt', 'checkpoint_last.pt', 'checkpoint90.pt', 'checkpoint100.pt'}
checkpoint_files = ['./checkpoints/' + folder for folder in os.listdir('checkpoints')]
for folder in checkpoint_files:
    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        whole_path = f'{folder}/{subfolder}'

        if 'checkpoint100.pt' not in os.listdir(whole_path):
            print('this folder does not contain checkpoint100.pt')
            print(whole_path)
            print(os.listdir(whole_path))
        for f in os.listdir(whole_path):
            whole_whole_path = whole_path + "/" + f
            if f in dont_delete_these:
                print(f'wont delete: {whole_whole_path}')
            else:
                print(f'deleting: {whole_whole_path}')
                os.remove(whole_whole_path)
