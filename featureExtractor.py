import glob
import os
import sys
from pathlib import Path

from sklearn import preprocessing
import datetime
from PIL import Image
import numpy as np
import cv2
import pandas as pd

def loadImageDataSet(dataset):
    """
    Based on the example given in TUWEL. Who am I kidding I copied the whole thing and adjusted it accordingly.
    :return:
    """
    # path to our image folder
    # For the first run, download the images from http://data.vicos.si/datasets/FIDS30/FIDS30.zip,
    # and unzip them to your folder
    #imagePath = Path("../data/raw/"+dataset)
    if dataset == 'cars':
        imagePath = Path("data/CarData/traintestImages")
        os.chdir(imagePath)
    else:
        imagePath = Path("data/FIDS30/")
        os.chdir(imagePath.absolute())

    # Find all images in that folder; there are like 1.000 different ways to do this in Python, we chose this one :-)

    if dataset == 'cars':
        fileNames = glob.glob("*/*.pgm")
    else:
        fileNames = glob.glob("*/*.jpg")

    numberOfFiles = len(fileNames)
    targetLabels = []

    print("Found " + str(numberOfFiles) + " files\n")

    # The first step - create the ground truth (label assignment, target, ...)
    # For that, iterate over the files, and obtain the class label for each file
    # Basically, the class name is in the full path name, so we simply use that
    for fileName in fileNames:
        if dataset == 'cars':
            pathSepIndex = fileName.index("-")
        else:
            pathSepIndex = fileName.index("\\")
        targetLabels.append(fileName[:pathSepIndex])

    # sk-learn can only handle labels in numeric format - we have them as strings though...
    # Thus we use the LabelEncoder, which does a mapping to Integer numbers
    le = preprocessing.LabelEncoder()
    le.fit(targetLabels)  # this basically finds all unique class names, and assigns them to the numbers
    print("Found the following classes: " + str(list(le.classes_)))

    # now we transform our labels to integers
    target = le.transform(targetLabels)
    print('Transformed labels (first elements: ' + str(target[0:150]))

    # If we want to find again the label for an integer value, we can do something like this:
    # print list(le.inverse_transform([0, 18, 1]))

    print("... done label encoding")

    # so NOW we actually extract features from our images

    print("Extracting features using PIL/PILLOW ({})".format(str(datetime.datetime.now())))

    # The simplest approach is via the PIL/PILLOW package; here we get a histogram over each RGB channel
    # Note: this doesn't really represent colours, as a colour is made up of the combination of the three channels!
    # Fore more information: https://stackoverflow.com/questions/22460577/understanding-histogram-in-pillow
    # Basically you only get the value for one channel and don't care about the others,
    # otherwise you'd have 256^3 combinations
    data = []
    for index, fileName in enumerate(fileNames):
        imagePIL = Image.open(os.getcwd() + '/' + fileName)
        # Not all images in our dataset are in RGB color scheme (e.g. indexed colours)
        # We need to make sure that they are RGB , otherwise we can't expect to have exactly three RGB channels..
        imagePIL = imagePIL.convert('RGB')
        featureVector = imagePIL.histogram()

        if len(featureVector) != 768:  # just a sanity check; with the transformation to RGB, this should never happen
            print("Unexpected length of feature vector: {} in file: ".format(str(len(featureVector)), fileName))

        data.append(featureVector)

    # Next, we extract a few more features using OpenCV

    print("Extracting features using OpenCV ({})".format(str(datetime.datetime.now())))
    dataOpenCV_1D = []
    dataOpenCV_2D = []
    dataOpenCV_3D = []

    # use our own simple function to flatten the 2D arrays
    flatten = lambda l: [item for sublist in l for item in sublist]

    for fileName in fileNames:

        # the easiest way would to do the following:
        # imageOpenCV = cv2.imread(imagePath + fileName)

        # However, we have the same issue as before, and it is more difficult in OpenCV to convert to an RGB image
        # Thus we do this using PIL, and then convert to OpenCV ....
        imagePIL = Image.open(os.getcwd() + '/' + fileName)
        imagePIL = imagePIL.convert('RGB')
        imageOpenCV = np.array(imagePIL)
        # Convert RGB to BGR
        imageOpenCV = imageOpenCV[:, :, ::-1].copy()

        # Now we split the image in the three channels, B / G / R
        chans = cv2.split(imageOpenCV)
        colors = ("b", "g", "r")

        # First we do also features per channel, but this time, we aggregate them into a smaller number of bins
        # I.e. we do not have 256 values per channel, but less
        featuresOpenCV_1D = []
        bins_1D = 64
        for (chan, color) in zip(chans, colors):  # we compute the histogram over each channel
            histOpenCV = cv2.calcHist([chan], [0], None, [bins_1D], [0, 256])
            featuresOpenCV_1D.extend(histOpenCV)
        featureVectorOpenCV_1D = flatten(featuresOpenCV_1D)  # and append this to our feature vector

        dataOpenCV_1D.append(featureVectorOpenCV_1D)  # now we append the feature vector to the dataset so far

        if len(featureVectorOpenCV_1D) != bins_1D * 3:  # sanity check, in case we had a wrong number of channels...
            print("Unexpected length of feature vector: {} in file: {}".format(str(len(featureVectorOpenCV_1D)),
                                                                               fileName))

        # Next - features that look at two channels at the same time
        # E.g. we look at when green and blue have both "high values"
        # We reduce the size of bins further, to not have a too long feature vector
        featuresOpenCV_2D = []
        bins2D = 16
        # look at all combinations of channels (R & B, R & G, B & G)
        featuresOpenCV_2D.extend(cv2.calcHist([chans[1], chans[0]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
        featuresOpenCV_2D.extend(cv2.calcHist([chans[1], chans[2]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
        featuresOpenCV_2D.extend(cv2.calcHist([chans[0], chans[2]], [0, 1], None, [bins2D, bins2D], [0, 256, 0, 256]))
        # and add that to our dataset
        featureVectorOpenCV_2D = flatten(featuresOpenCV_2D)
        dataOpenCV_2D.append(featureVectorOpenCV_2D)

        # finally, we look at all three channels at the same time.
        # We further reduce our bin size, because otherwise, this would become very large...
        featuresOpenCV_3D = cv2.calcHist([imageOpenCV], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # append to our dataset
        featureVectorOpenCV_3D = featuresOpenCV_3D.flatten()
        dataOpenCV_3D.append(featureVectorOpenCV_3D)

    print(".... done ({})".format(str(datetime.datetime.now())))

    # And now we finally classify
    # these are our feature sets; we will use each of them individually to train classifiers
    trainingSets = [data, dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D]

    # Save data to CSV
    print('Saving data to CSV file formats ({}).'.format(str(datetime.datetime.now())))
    #data_path = Path('../data/interim/'+dataset)
    if dataset == 'cars':
        data_path = Path("C:/Princi/TU Wien/Semestri 1/Machine Learning/Exercises/Exercise 3/ml_exercise_3/main/data/Interim/cars")
    else:
        data_path = Path("C:/Princi/TU Wien/Semestri 1/Machine Learning/Exercises/Exercise 3/ml_exercise_3/main/data/Interim/fruits")

    for index, trainSet in enumerate(trainingSets):
        df = pd.DataFrame(trainSet)

        # Add class labels
        df['class'] = target

        # Naming convention
        if index == 0:
            df.to_csv(data_path / 'data.csv')
        else:
            df.to_csv(data_path / 'dataOpenCV_{}D.csv'.format(index))

    print('... done with saving CSVs ({}).'.format(str(datetime.datetime.now())))


def main():
    loadImageDataSet('cars')
    #loadImageDataSet('fruits')


if __name__ == '__main__':
    main()
    sys.exit(0)
