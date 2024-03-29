'''Created with python 3.8 and tensorflow 2.4
This requires the installation of the librosa library that cannot properly function without the FFmegg packages
Read the songs in each folders and return as many pictures as may be produced using the windows size (sampling) and the amont of overlapping (overlap)'''

from os import listdir
from os.path import isfile, join, isdir

import numpy as np

import librosa
import librosa.display

import matplotlib.pyplot as plt


path_folder = 'genre/'

overlap = 0.1
sampling = 0.5

folderList = [f for f in listdir(path_folder) if isdir(join(path_folder, f))]

for folder in folderList:
    pathFileGenre = path_folder + folder + '/'
    filesList = [f for f in listdir(pathFileGenre) if isfile(join(pathFileGenre, f))]
    print(folder)
    Infolder = 'picture_bigger_windows/' + folder + '/'
    filesInFolder = [f for f in listdir(Infolder) if isdir(join(Infolder, f))]
    for file in filesList:
        if file not in filesInFolder:
            path_file = pathFileGenre + file
            raw_audio, rate = librosa.load(path_file)
            raw_audio = raw_audio[:660000]

            # splitting of the songs
            shape = raw_audio.shape[0]
            samp_box = int(shape * sampling)
            offset = int(shape * sampling * (1. - overlap))
            songs = [raw_audio[i:i + samp_box]
                     for i in range(0, shape - samp_box + offset, offset)]

            count = 0
            for song in songs:
                #Use the mel spectrogram function to return the fourier transform of the song sample.
                spect = librosa.feature.melspectrogram(song, n_fft=1024,
                                                       hop_length=256,
                                                       n_mels=128)

                if file.endswith('.mp3'):
                    filename = file[:-4]
                    filename = str(count) + filename
                count = count + 1

                ax = plt.axes()
                ax.set_axis_off()
                plt.set_cmap('hot')
                librosa.display.specshow(spect, y_axis='log', x_axis='time', sr=rate)
                path_save = 'picture_bigger_windows/' + folder + '/' + filename + '.png'
                #Save each song samples in a new folder corresponding to the original genre
                plt.savefig(path_save)
