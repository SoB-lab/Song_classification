from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
import shutil, os


mypath = 'picture_database/'
new_path = 'picture_database_seperated/'
ListFolders = [g for g in listdir(mypath) if isdir(join(mypath, g))]
for folder in ListFolders:
    print(folder)
    mypath_f = mypath + folder + '/'
    onlyfiles = [f for f in listdir(mypath_f) if isfile(join(mypath_f, f))]
    nb_files = round(len(onlyfiles) * 0.1)
    for filename in onlyfiles[:nb_files]:
        file = mypath_f + filename
        end_path = new_path + 'test/' + folder + '/'
        shutil.copy(file, end_path)
    for filename in onlyfiles[nb_files:]:
        file = mypath_f + filename
        end_path = new_path + 'train/' + folder + '/'
        shutil.copy(file, end_path)

