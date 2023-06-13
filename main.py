from matplotlib import pyplot as plt
import cv2

from utils.deskew_plate import deskew
from license_plate_recognition import segment_characters, show_results, load_char_recog_model
from license_plate_detection import getNumberPlateRegion


def ANPR(img_path):
    img = cv2.imread(img_path)
    cv2.imshow("Input_Image", img)
    np_region = getNumberPlateRegion(img_path)

    if np_region is None:
        print("No number plate found")
        return None
    else:

        np_region = cv2.resize(np_region,
                               None,
                               fx=2,
                               fy=2,
                               interpolation=cv2.INTER_CUBIC)
        np_region = cv2.fastNlMeansDenoisingColored(np_region, None, 10, 10, 7,
                                                    15)

        cv2.imshow("Number plate", np_region)
        np_region = deskew(np_region)
        cv2.imshow("Tilt fixed Number plate", np_region)
        char = segment_characters(np_region)
        num_plate = show_results(char)

        # For showing results
        print("Number plate detected: " + num_plate)
        plt.figure(figsize=(10, 6))
        for i, ch in enumerate(char):
            img = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
            plt.subplot(3, 4, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f'predicted: {num_plate[i]}')
            plt.axis('off')
        plt.savefig("mygraph.png")
        mygraph = cv2.imread("mygraph.png")
        cv2.imshow("Results", mygraph)
        cv2.waitKey(0)
        return num_plate


if __name__ == '__main__':
    img_path = f"D:/ANPR/ANPR-ML/Sample Images/2.jpg"
    print(ANPR(img_path))