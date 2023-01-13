
import os
import random
import numpy as np


data_folder = './omniglot_resized'
character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]


sampler = lambda x: random.sample(x, 3)

characters = np.random.choice(character_folders, size=3, replace=False)

labels = [os.path.basename(os.path.dirname(c)) +'-'+ os.path.basename(c) for c in character_folders]

# images_labels = [(i, os.path.join(path, image))
                 
#                      for i, path in zip(labels, characters)
#                      for image in sampler(os.listdir(path))]

# for image in sampler(os.listdir(characters[0])):
#     print(image)
# print(images_labels)
print(os.listdir(characters[0]))
for i in os.listdir(characters[0]):
    print(i.decode('utf-8'))