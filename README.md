# Song_classification

Use the FMA database to create a genre classification tool using deep learning (CNN model) by converting the songs to pictures and applying Fourier transform.

This program is adapted for the classification of 3 music genre from the GTZAN FFMA datasets. 


To obtain everything right it should be run as follow:
1) new_database.py
2) seperation_database.py: creates a seperated database for 10% test sample and 90% training data (the file are not yet pull randomly)
3) main.py: Generate the datasets and run the model. The model is saved in 'my_model' for testing and analysis. And create an history.csv file that returns the loss and accuracy of both the training and the validation datasets.
4) extension_training.py: extend the training by loading the saved model and extend the number of epochs. The results are stored in a history.csv and the new model is saved in 'my_model'.
5) test_model.py: Tests the model saved in 'my_model' with the test dataset. Returns the results of the accuracy on the tests and the confusion matrix. 



Additional files: conversion_to_pictures.py For entire generation of the entire picture corresponding to the mel spectrogram (no sampling)
