'''
    File name: TVD.py
    Copyright: Copyright 2017, The CogWorks Laboratory
    Author: Hassan Alshehri
    Email: alsheh@rpi.edu or eng.shehriusa@gmail.com
    Data created: May 22, 2017
    Date last modified: August 26, 2017
    Description: Character Recognition
    Status: Research
    Requirements/Dependencies:
        1. Python 2.7,
        2. OpenCV 3 (may not work with OpenCV 2),
        3. NumPy
'''
import cv2
import numpy as np
import operator
import os
import argparse
import re

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def removeBackground(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of background color in HSV
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, 90])

    # Create a mask of the background.
    # This makes the background white and everything else black.
    mask = cv2.inRange(hsv, lower, upper)

    # Inverse the mask of the background to make
    # the backgorund black and everything else white.
    maskInv = cv2.bitwise_not(mask)

    # Remove the background from the color image
    colorImg = cv2.bitwise_and(img, img, mask=maskInv)

    return colorImg, maskInv


class OCR:
    '''
    This class implments character recognition using K nearest neighbors (KNN) algorithm.
    '''
    def __init__(self):
        # Read in training classifications KNN
        try:
            path = re.sub('ocr.*', 'classifications.txt', __file__)
            npaClassifications = np.loadtxt(path, np.float32)
        except:
            print "error, unable to open classifications.txt, exiting program\n"
            os.system("pause")
            exit(1)

        # Read in training images
        try:
            path = re.sub('ocr.*', 'flattened_images.txt', __file__)
            npaFlattenedImages = np.loadtxt(path, np.float32)
        except:
            print "error, unable to open flattened_images.txt, exiting program\n"
            os.system("pause")
            exit(1)

        # Reshape numpy array to 1d, necessary to pass to call to train
        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

        # Instantiate KNN object
        self.kNearest = cv2.ml.KNearest_create()

        self.kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

        self.DIGIT_LOW = [0, 0, 0] # default [0, 0, 0]
        self.DIGIT_HIGH = [179, 255, 100] # default [179, 255, 100]


    def readCharacters(self, imgTestingNumbers, isWhiteBackground=False, minArea=0,game=None, firstChar=None):
        # This list will store cropped characters from the image
        validContoursWithData = []


        # Resize image for consistency to avoid too small or too large images.
        w, h = imgTestingNumbers.shape[1::-1]
        scale = float(135)/h
        imgTestingNumbers = cv2.resize(imgTestingNumbers, (int(w*scale), int(h*scale)))


        # If the the image has a white background, make backgroudn black and rest of image white
        if isWhiteBackground:
            imgTestingNumbers = cv2.threshold(imgTestingNumbers, 200, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            w, h = imgTestingNumbers.shape[1::-1]
            minArea = h * h * 0.1
        
        # Remove background and noise from image
        colorImg, imgThresh = removeBackground(imgTestingNumbers)
        cv2.imshow("thresh", imgThresh)
        
        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
        imgThreshCopy = imgThresh.copy()
        # arg#2: retrieve the outermost contours only
        # arg#3: compress horizontal, vertical, and diagonal segments and leave only their end points
        _, contours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        # Iterate over all contours (digits) and reject too small or large ones.
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w*h < minArea:
                continue
            contour = imgTestingNumbers[y:y+h, x:x+w]
            cv2.rectangle(imgTestingNumbers, (x, y), (x+w, y+h), (180,180,0), 2)
            validContoursWithData.append( (x,y,w,h) )
        # Sort contours from left to right
        validContoursWithData.sort()
        
        # Declare final string, this will have the final number sequence by the end of the program.
        strFinalString = ""
        # Run the KNN algorithm on each contour.
        for c in validContoursWithData:
            # Draw a green rect around the current char
            x, y, w, h = c

            # Crop char out of threshold image
            imgROI = imgThresh[y:y+h, x:x+w]

            # Resize image, this will be more consistent for recognition and storage
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            # Flatten image into 1d numpy array
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

            # Convert from 1d numpy array of ints to 1d numpy array of floats
            npaROIResized = np.float32(npaROIResized)

            # Call KNN function to find nearest character
            retval, npaResults, neigh_resp, dists = self.kNearest.findNearest(npaROIResized, k = 1)

            # Reject the character if its not within the acceptable limit.
            if dists[0][0]/1000000.0 > 11.0:
                # Draw rectangle on original testing image
                print "====>", str(chr(int(npaResults[0][0])))
                cv2.rectangle(imgTestingNumbers, (x, y), (x+w, y+h), (0, 0, 255), 2)
                #print str(chr(int(npaResults[0][0]))), dists[0][0]/1000000.0
                continue

            # Draw rectangle on original testing image
            cv2.rectangle(imgTestingNumbers, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Get character from results
            strCurrentChar = str(chr(int(npaResults[0][0])))

            # Append current char to full string
            strFinalString = strFinalString + strCurrentChar

        return strFinalString, imgTestingNumbers, imgThresh



    def testOCR(self):
        LABELS = ['0123456789', '574920', '027430', '602350', '027430', '166', '059',\
           '22', '015', '647500', '034790', '682290', '034790', '203', '26', '072', '665240', \
           '722990', '27', '076', '076', '745780', '018250', '764030', '082', '223', '228', '082',\
           '083', '572160', '019610', '591770', '159', '161', '21', '058', '159', '058', '542580', \
           '154', '563610', '153', '000001', '007', '096900', '106852', '014', '036', '013', '471960',\
           '087075', '111', '087075', '609300', '007214', '056', '007088', '212172', '143', '143', '20',\
           '101548', '513100', '147', '052', '033710', '204', '6503428']

        totalTests = 68
        totalFail = 0
        for i in range(totalTests):
            filename = __file__.replace('ocr.py', 'test_cases/')
            if i < 10:
                filename += '0'+str(i)
            else:
                filename += str(i)
            filename += '.png'

            # Read in testing numbers image
            img = cv2.imread(filename)

            # If image was not read successfully, report error and exit
            if img is None:
                print "error: image not read from file \n\n"
                os.system("pause")
                exit(1)

            if i == 67:
                resultString, resultImg, imgThresh = self.readCharacters(img, game=self, isWhiteBackground=True)

            else:
                resultString, resultImg, imgThresh = self.readCharacters(img, game=self)

            print "[test#%s]" %(str(i).rjust(2, '0')),
            if resultString == LABELS[i]:
                print "PASS",
            else:
                totalFail += 1
                print "FAIL",

            print "result:[%s]" %resultString, "expected:[%s]" %LABELS[i]
            print "Press any key to test next image\n"
            cv2.imshow("resultImg", resultImg)
            cv2.waitKey(0)

        print "-" * 40
        print "PASSED: %d tests" %(totalTests-totalFail)
        print "FAILED: %d tests" %totalFail

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Character Recognition Program')

    # input image
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='path to input image file.')

    parser.add_argument('-i', '--img-path',
                        help='path to input image file (use quotes to include spaces in path).')

    parser.add_argument('-w','--white-background',
                        action='store_true',
                        help='input image has white background (default: %(default)s).')

    # Start parsing command line arguments
    args = parser.parse_args()

    # create OCR object
    ocr = OCR()

    if args.test:
        ocr.testOCR()
    elif args.img_path is not None:
        imgPath = args.img_path
        isWhiteBackground = args.white_background

        # read in testing numbers image
        img = cv2.imread(imgPath)

        # if image was not read successfully, report error and exit
        if img is None:
            print "error: image not read from file \n\n"
            os.system("pause")
            exit(1)

        resultString, resultImg, imgThresh = ocr.readCharacters(img, isWhiteBackground=isWhiteBackground, game=ocr)

        print "result:[%s]" %resultString
        print "Press any key to close the window"
        cv2.imshow("resultImg", resultImg)
        cv2.waitKey(0)
    else:
        print "Please use the appropriate flags to run the program"
