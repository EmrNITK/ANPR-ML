from matplotlib import pyplot as plt
import cv2
import numpy as np

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential

CHAR_RECOGNITION_YOLO_MODEL_PATH = "D:/ANPR/ANPR-ML/checkpoints/checkpoints/my_checkpoint"


# Match contours to license plate or character template
def find_contours(dimensions, img):

    # cv2.imshow("before img", img)
    # cv2.waitKey(0)

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours
    # -1 signifies drawing all contours
    # cv2.drawContours(img, cntrs, -1, (0, 255, 0), 2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    # print(dimensions)
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        # print(cv2.boundingRect(cntr))

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(
                intX
            )  #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY),
                          (intWidth + intX, intY + intHeight), (50, 21, 200),
                          2)
            plt.imshow(ii, cmap='gray')
            plt.title('Predict Segments')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(
                char_copy
            )  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    # plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(
            img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


# Find characters in the resulting images
def segment_characters(image):

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray_lp, (7, 7), 0)
    img_binary_lp = cv2.adaptiveThreshold(img_gray_lp, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 51, 10)
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(img_binary_lp, cv2.MORPH_OPEN, kernel)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [
        LP_WIDTH / 6, LP_WIDTH / 2, LP_HEIGHT / 10, 2 * LP_HEIGHT / 3
    ]
    # plt.imshow(img_binary_lp, cmap='gray')
    # plt.title('Contour')
    # plt.show()
    cv2.imwrite('contour.jpg', img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)
    return char_list


def load_char_recog_model():
    # Create a new model instance
    loaded_model = Sequential()
    loaded_model.add(
        Conv2D(16, (22, 22),
               input_shape=(28, 28, 3),
               activation='relu',
               padding='same'))
    loaded_model.add(
        Conv2D(32, (16, 16),
               input_shape=(28, 28, 3),
               activation='relu',
               padding='same'))
    loaded_model.add(
        Conv2D(64, (8, 8),
               input_shape=(28, 28, 3),
               activation='relu',
               padding='same'))
    loaded_model.add(
        Conv2D(64, (4, 4),
               input_shape=(28, 28, 3),
               activation='relu',
               padding='same'))
    loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
    loaded_model.add(Dropout(0.4))
    loaded_model.add(Flatten())
    loaded_model.add(Dense(128, activation='relu'))
    loaded_model.add(Dense(36, activation='softmax'))

    # Restore the weights
    loaded_model.load_weights(
        CHAR_RECOGNITION_YOLO_MODEL_PATH
    ).expect_partial()
    return loaded_model


# Predicting the output
def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
        return new_img


def show_results(model,char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c
    loaded_model = model
    output = []
    for i, ch in enumerate(char):  #iterating over the characters
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  #preparing image for the model
        predict_x = loaded_model.predict(img)
        y_ = np.argmax(predict_x, axis=1)[0]
        character = dic[y_]
        output.append(character)  #storing the result in a list

    plate_number = ''.join(output)

    return plate_number
