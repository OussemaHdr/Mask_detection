import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
data = pd.read_csv("train_labels.csv")

pictures_folder = ".."
target = ".."

#puting every picture in its associated class folder
for item in tqdm(range(0, data.shape[0])):
    try :
        picture_name = data['image'][item]
        l = picture_name.split('.')
        picture = Image.open(pictures_folder  + l[0]+'.png')
        picture_class = data['target'][item]
        picture.save(target + str(picture_class) + '/' + l[0]+'.png')
    except :
        print('bug')
