import pandas as pd
import os
import numpy as np
import shutil

dataset = pd.read_csv('all_data_info.csv')
file_names = list(dataset['new_filename'].values)
img_labels = list(dataset['artist'].values)

folders_to_be_created = np.unique(list(dataset['artist'].values))

source = os.getcwd()

for new_path in folders_to_be_created:
    if not os.path.exists(new_path):
        os.makedirs(new_path)



folders = folders_to_be_created.copy()

for f in range(len(file_names)):
	current_img = file_names[f]
	current_label = img_labels[f]
	if os.path.exists(current_img):
		shutil.move(current_img, current_label)