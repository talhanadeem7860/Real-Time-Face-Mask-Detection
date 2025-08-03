# Real-Time-Face-Mask-Detection
This project is a real-time computer vision application designed to detect faces in a video stream and classify them based on whether they are wearing a face mask. It uses a deep learning model built with TensorFlow/Keras on top of a pre-trained face detector from OpenCV.

The system follows a two-stage process:
Face Detection: A highly optimized, pre-trained SSD (Single Shot Detector) model locates all faces in the current frame.
Mask Classification: A custom-trained classifier, fine-tuned using the MobileNetV2 architecture, analyzes each detected face and classifies it as With Mask or Without Mask.

Key Features:

Real-time face detection from a live webcam feed.

Classification of faces into With Mask and Without Mask.

Uses a deep learning model fine-tuned with transfer learning for high accuracy.

Visual feedback with color-coded bounding boxes and confidence scores.

Methodology
The project is executed in two main phases:

Model Training: A custom classifier is trained using transfer learning on the MobileNetV2 architecture. The model learns to distinguish between faces with and without masks from a large public dataset (Face Mask 12k Images Dataset from Kaggle). The final trained model is then saved for inference.

Real-Time Inference: A second script accesses the live webcam feed. It uses a pre-trained face detector from OpenCV to locate faces in each frame. Each detected face is then passed to the custom-trained classifier, which returns a prediction. The result is visualized in real-time with a colored bounding box and a confidence score.

