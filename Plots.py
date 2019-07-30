# import the necessary packages
import sys
from matplotlib import pyplot as plt
# import libraries for image feature extraction
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def featureExtractionVisualizationFruit():

    p = Path('C:/Princi/TU Wien/Semestri 1/Machine Learning/Exercises/Exercise 3 - New/ML_Exercise3/data/FIDS30/oranges/5.jpg')
    dImage = str(p)


    imagePIL = Image.open(dImage)
    imgplot = plt.imshow(imagePIL)
    plt.title("oranges/1.jpg")

    featureVector = imagePIL.histogram()

    # We plot this histogram
    plt.figure()
    plt.plot(featureVector[:256], 'r')
    plt.plot(featureVector[257:512], 'g')
    plt.plot(featureVector[513:], 'b')
    plt.xlim([0, 256])
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    plt.title("Colour Histogram - PIL")

    imageOpenCV = cv2.imread(dImage)

    plt.figure()
    plt.imshow(cv2.cvtColor(imageOpenCV, cv2.COLOR_BGR2RGB))

    chans = cv2.split(imageOpenCV)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Colour Histogram - OpenCV")
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    featuresOpenCV = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and add it to the resulting histograms array (of arrays)
        # We can specifiy here in the 4th argument how many bins we want -
        # 256 means the same as in the previous histogram
        histOpenCV = cv2.calcHist([chan], [0], None, [256], [0, 256])
        featuresOpenCV.extend(histOpenCV)

        # plot the histogram of the current colour
        plt.plot(histOpenCV, color=color)
        plt.xlim([0, 256])

    # Now we have a 2D-array - 256 values for each of 3 colour channels.
    # To input this into our machine learning, we need to "flatten" the features into one larger 1D array
    # the size of this will be 3 x 256 = 768 values
    featureVectorOpenCV = np.array(featuresOpenCV).flatten()

    # show all the plots
    plt.show()


def featureExtractionVisualizationCar():

    p = Path('C:/Princi/TU Wien/Semestri 1/Machine Learning/Exercises/Exercise 3 - New/ML_Exercise3/data/CarData/TrainImages/pos-0.pgm')
    dImage = str(p)


    imagePIL = Image.open(dImage)
    imgplot = plt.imshow(imagePIL)
    plt.title('pos-0.pgm')


    featureVector = imagePIL.histogram()

    # We plot this histogram
    plt.figure()
    plt.plot(featureVector[:256], 'r')
    plt.plot(featureVector[257:512], 'g')
    plt.plot(featureVector[513:], 'b')
    plt.xlim([0, 256])
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    plt.title("Colour Histogram - PIL")

    # plt.show()

    # An alternative is to use open CV
    imageOpenCV = cv2.imread(dImage)


    plt.figure()
    plt.imshow(cv2.cvtColor(imageOpenCV, cv2.COLOR_BGR2RGB))

    chans = cv2.split(imageOpenCV)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Colour Histogram - OpenCV")
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    featuresOpenCV = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):

        histOpenCV = cv2.calcHist([chan], [0], None, [256], [0, 256])
        featuresOpenCV.extend(histOpenCV)

        # plot the histogram of the current colour
        plt.plot(histOpenCV, color=color)
        plt.xlim([0, 256])

    # Now we have a 2D-array - 256 values for each of 3 colour channels.
    # To input this into our machine learning, we need to "flatten" the features into one larger 1D array
    # the size of this will be 3 x 256 = 768 values
    featureVectorOpenCV = np.array(featuresOpenCV).flatten()

    # show all the plots
    plt.show()


def main():
    """
    Main function.
    :return:
    """
    featureExtractionVisualizationFruit()
    featureExtractionVisualizationCar()


if __name__ == '__main__':
    main()
    sys.exit(0)

