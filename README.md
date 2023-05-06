



<!-- PROJECT LOGO -->
<br />
<div align="center">
<p align="center">
  <img src="Illustrations/ANPR_BANNER.png" />
</p>
 

  <h1 align="center">ANPR System</h1>

  <p align="center">
Automatic Number Plate Recognition (ANPR) is a computer vision technology that enables automatic detection, reading, and recognition of license plate numbers on vehicles.
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#approach">Approach</a></li>
    <li><a href="#datasets">Datasets</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com)-->

An Automatic Number Plate Recognition (ANPR) system is a computer vision technology to read and recognize license plate numbers on vehicles. ANPR systems are commonly used for security and surveillance purposes, such as identifying stolen or wanted vehicles, monitoring traffic flow, and enforcing parking regulations.

Implementing an ANPR system involves several steps, including image acquisition, preprocessing, segmentation and character recognition. The system captures an image of the vehicle's license plate using a camera, applies various image processing techniques to enhance the image quality and extract the license plate region, segments the individual characters on the plate, recognizes the characters.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

For testing the ANPR system on sample images, run main.py file.
  ```sh
  python main.py
  ```

### Installation

To install these tools, follow the instructions below:

## Python: 
Download and install the latest version of Python from the official website: https://www.python.org/downloads/

## OpenCV: 
Install OpenCV using the following command in your terminal or command prompt: 
```sh 
pip install opencv-python 
```

## YOLOv7: 
Clone the YOLOv7 repository from GitHub: https://github.com/WongKinYiu/yolov7.git 

## Torch: 
Install PyTorch using the following command:
```sh 
pip install torch 
```

## Numpy: 
Install Numpy using the following command:
```sh 
pip install numpy 
```

## Tensorflow:
Install Tensorflow using the following command:
```sh 
pip install tensorflow 
```

## Scikit-learn: 
Install Scikit-learn using the following command: 
```sh 
pip install scikit-learn 
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Approach

Our algorithm works as follows:
1. The license plate is detected from the input image. It has been done using an object detection method: You-Only-Look-Once(YOLOv7).
2. Once we get the number plate region, Deskewing is done for correcting the rotated number plates.
3. Character segmentation is done to extract the characters from the number plate using adaptive thresholding and contour manipulation.
4. At last step, segmented characters are recognized using CNN.

<br />
<div align="center">
<p align="center">
  <img src="Illustrations/anpr%20flowchart.png" width=640/>
</p>
</div>

<!-- Datasets -->
## Datasets

The following datasets have been used in this ANPR system:
<br />
## For License plate detection (YOLOv7): 
This dataset contains 453 files - images in JPEG format with bounding box annotations of the car license plates within the image. Annotations are provided in the PASCAL VOC format. Pascal VOC(Visual Object Classes) is a format to store annotations for localizer or Object Detection datasets and is used by different annotation editors and tools to annotate, modify and train Machine Learning models. The dataset can be found and downloaded from here https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection
You can download trained model from here: https://drive.google.com/file/d/1f7tWbDNqfWqca1gy91ItV5gkwUOpLAHx/view?usp=sharing
<br />
## For Character recognition: 
The dataset has about 2000 images of digits from 0-9 and alphabets from A-Z. You can download the dataset from here: https://drive.google.com/file/d/1y8OG1GwfTd7fZMpu7rFsSVEa3wj1I98q/view?usp=sharing
<br />
## For testing the whole model: 
The dataset contains about 166 images of cars with license plates. You can find the dataset here: https://drive.google.com/file/d/1r_X_Wf4FQ8AcgOvy6o2AdBk11LJB67D_/view?usp=sharing
<br />

<!-- Results -->
## Results
 License plate detection using YOLOv7 works perfectly on most images. It fails in some cases when a vehicle contains text similar to number plate.
 
 The accuracy of CNN used for character recognition is about 98%.
 
 The overall accuracy of the ANPR system proposed here is 87.75%, which is tested on test_dataset mentioned in dataset section.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


