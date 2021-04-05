'''
For each folder in genre, create a genre folder in picture dataset
For each file in each folder read the audio file, convert to picture, save in picture folder
'''

import librosa
import librosa.display
import matplotlib.pyplot as plt
from os.path import isfile, join, isdir
from os import listdir

path_data = 'genre/'
folderList = [f for f in listdir(path_data) if isdir(join(path_data, f))]

#for folders in folderList:
folders = 'Rock'
print('Starting with first folder')
print(folders)
pathFileGenre = 'genre/' + folders + '/'
filesList = [f for f in listdir(pathFileGenre) if isfile(join(pathFileGenre, f))]
for files in filesList:
    pathFile = pathFileGenre + '/' + files
    if files.endswith('.mp3'):
        filename = files[:-4]
    path_save = 'picture_database/' + folders +'/'+ filename + '.png'
    raw_audio, rate = librosa.load(pathFile, sr=None, mono=True)

    # Get fourier transform picture
    stft = librosa.stft(raw_audio, n_fft=2048, hop_length=512)  # Get the fourier transform
    Xdb = librosa.amplitude_to_db(abs(stft))
    plt.figure(figsize=(12, 4))
    ax = plt.axes()
    ax.set_axis_off()
    plt.set_cmap('hot')
    librosa.display.specshow(Xdb, y_axis='log', x_axis='time', sr=rate)
    plt.savefig(path_save)
