# Face_Recognition_Using_MTCNN
This repository contains a face recognition model implemented using MTCNN for face detection and a deep learning model for face recognition. MTCNN is a popular method for detecting faces in images and videos with high accuracy, and it is combined with a pre-trained model (such as Facenet, VGGFace, or ResNet) for face recognition.

Table of Contents
Overview
Features
Installation
Usage
Dataset
Model Training
Face Detection and Recognition
Results
Contributing
License
Overview
This project aims to build a face recognition system that can:

Detect faces in an image or video.
Recognize detected faces and label them correctly based on a pre-trained model.
The project utilizes MTCNN for face detection and a deep learning model such as Facenet for face recognition. MTCNN is particularly effective for detecting faces in various orientations and lighting conditions.

Features
Face Detection: Efficient and accurate detection of faces using MTCNN.
Face Recognition: Recognition and labeling of detected faces using a pre-trained deep learning model.
Support for Images and Videos: Works with both static images and live video streams.
Real-Time Processing: Capable of performing real-time face recognition with decent processing power.
Installation
To use this project, you need to have Python installed along with the necessary libraries. Follow these steps to set up the environment:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/face-recognition-mtcnn.git
cd face-recognition-mtcnn
Install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
The requirements.txt file includes the following libraries:

tensorflow or pytorch
keras
mtcnn
opencv-python
numpy
scipy
Usage
To perform face detection and recognition:

Detect Faces in an Image:

python
Copy code
python detect_faces.py --image_path ./data/sample_image.jpg
Recognize Faces in an Image:

python
Copy code
python recognize_faces.py --image_path ./data/sample_image.jpg
Recognize Faces in a Video:

python
Copy code
python recognize_faces_video.py --video_path ./data/sample_video.mp4
Dataset
For training and testing, you can use publicly available datasets like:

Labeled Faces in the Wild (LFW)
VGGFace2
Ensure that the dataset is downloaded and organized in the proper directory structure for the model to use.

Model Training
To train the face recognition model, you can use the train_model.py script. Make sure to update the script with the path to your dataset and desired parameters:

bash
Copy code
python train_model.py --dataset_path ./data/dataset --epochs 50 --batch_size 32
Face Detection and Recognition
The mtcnn library is used for face detection. Once the faces are detected, a deep learning model (such as Facenet) is used to recognize and label the faces.

Results
The results are displayed in terms of accuracy, precision, recall, and F1 score. Sample images and videos with detected and recognized faces are saved in the output/ folder.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue to improve the project.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

