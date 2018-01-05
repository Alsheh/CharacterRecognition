# GenData.py

import sys
import numpy as np
import cv2
import os
import argparse

# Module level variables
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main(imgTrainingNumbers):
    # If image was not read successfully, report error msg and exit
    if imgTrainingNumbers is None:
        print "error: image not read from file \n\n"
        os.system("pause")
        return

    # Get grayscale image
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    # Blur the gray image
    #imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)


    # filter image from grayscale to black and white
    # arg1: input image
    # arg2: make pixels that pass the threshold full white
    # arg3: use gaussian rather than mean, seems to give better results
    # arg4: invert so foreground will be white, background will be black
    # arg5: size of a pixel neighborhood used to calculate threshold value
    # arg6: constant subtracted from the mean or weighted mean
    #imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    imgThresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # Make a copy of the thresh image, this in necessary b/c findContours modifies the image
    imgThreshCopy = imgThresh.copy()

    # Show threshold image for reference
    #cv2.imshow("imgGray", imgGray)
    # Show threshold image for reference
    cv2.imshow("imgThresh", imgThresh)
    cv2.waitKey(0)

    # arg#1: input image, make sure to use a copy since the function will modify this image in the course of finding contours
    # arg#2: retrieve the outermost contours only
    # arg#3: compress horizontal, vertical, and diagonal segments and leave only their end points
    _, npaContours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Declare empty numpy array, we will use this to write to file later
    # zero rows, enough cols to hold all image data
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    # Declare empty classifications list, this will be our list of how we are classifying our chars
    # from user input, we will write to file at the end
    intClassifications = []

    #Ppossible chars we are interested in are digits 0 through 9, put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'), ord('a'), ord('b'), ord('c'), ord('d'),
                     ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'), ord('k'), ord('l'), ord('m'), ord('n'),
                     ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'), ord('u'), ord('v'), ord('w'), ord('x'),
                     ord('y'), ord('z')]


    # For each digit
    for npaContour in npaContours:
        # If contour is big enough to consider
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            # Get and break out bounding rect
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            # Draw rectangle around each contour as we ask user for input
            # Draw rectangle on original training image
            cv2.rectangle(imgTrainingNumbers, (intX, intY), (intX+intW,intY+intH), (0, 255, 0), 2)

            # Crop char out of threshold image
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]

            # Resize image, this will be more consistent for recognition and storage
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            # Show cropped out char for reference
            #cv2.imshow("imgROI", imgROI)
            # Show resized image for reference
            #cv2.imshow("imgROIResized", imgROIResized)
            # Show training numbers image, this will now have red rectangles drawn on it
            cv2.imshow("training_numbers.png", imgTrainingNumbers)

            # Get key press
            intChar = cv2.waitKey(0)

            # If esc key was pressed, exit program
            if intChar == 27:
                sys.exit()
            # Else if the char is in the list of chars we are looking for . . .
            elif intChar in intValidChars:
                # Append classification char to integer list of chars (we will convert to float later before writing to file)
                intClassifications.append(intChar)
                # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                # Add current flattened impage numpy array to list of flattened image numpy arrays
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
                cv2.rectangle(imgTrainingNumbers, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)

    # Convert classifications list of ints to numpy array of floats
    fltClassifications = np.array(intClassifications, np.float32)

    # Flatten numpy array of floats to 1d so we can write to file later
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print "\n\nTraining Complete!!\n"

    # Write flattened images to file
    with open('classifications.txt', 'a') as f_handle:
        np.savetxt(f_handle, npaClassifications)

    with open('flattened_images.txt', 'a') as f_handle:
        np.savetxt(f_handle, npaFlattenedImages)

    # Remove windows from memory
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Data Program')

    parser.add_argument('-i', '--img-path',
                        help='path to input image file (use quotes to include spaces in path).')

    # Start parsing command line arguments
    args = parser.parse_args()

    # Get image path
    imgTrain = args.img_path

    # Read in training numbers image
    img = cv2.imread(imgTrain)

    # Start training and generating data
    main(img)
