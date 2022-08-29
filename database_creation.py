# Import libraries
import os
from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
import shutil, os


# Define functions

# Start of the main program
mypath = 'fma_small/'
mypath_g = 'genre/'

file_tracks = 'fma_metadata/tracks.csv'
tracks = pd.read_csv(file_tracks, index_col = 0, header=[0, 1])
ext = '.mp3'

total = 0
ListFolders = [g for g in listdir(mypath) if isdir(join(mypath, g))]
for folder in ListFolders:
    mypath_f = mypath + folder + '/'
    onlyfiles = [f for f in listdir(mypath_f) if isfile(join(mypath_f, f))]
    total = total + len(onlyfiles)
    count = 0
    for files in onlyfiles:
        onlyfolders = [f for f in listdir(mypath_g) if isdir(join(mypath_g, f))]
        if files.endswith(ext):
            track_number = files[:-4]
        track_number = track_number.lstrip('0')
        index = int(track_number)
        genre = tracks.loc[[index]]
        genre_striped = str(genre.iloc[:, 39].values).strip('[\'\']')
        if genre_striped not in onlyfolders:
            path_new_folder = 'genre/' + str(genre_striped)
            os.makedirs(path_new_folder)
        path_for_copy = 'genre/' + str(genre_striped) + '/'
        file = mypath_f + files
        shutil.copy(file, path_for_copy)
print(total)





