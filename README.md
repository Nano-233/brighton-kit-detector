# **Custom Football Kit Detector**

A real-time object detection project that uses a custom-trained YOLO model to identify and classify specific football kits from a live webcam feed or video file.

---

## **Project Demo**

![Football Kit Detector Demo](img/output.gif)

---

## **Features**

* **Real-time Detection:** Identifies 6 different football kits from a live video feed.  
* **Custom-Trained Model:** Trained from scratch on a personally-collected dataset of hundreds of images.  
* **High Performance:** Uses the yolo11s model for a great balance of speed and accuracy.  
* **Customizable:** Easily configurable with custom bounding box colors and output resolutions.

---

## **The Pipeline**


### **1\. Data Collection & Labeling**

The biggest challenge was creating a robust dataset. I personally collected and manually labeled over hundreds of images.

* **Strategy:** The dataset includes a mix of photos with the shirts laid flat (to capture clean patterns) and photos of the shirts being worn (to train the model on real-world wrinkles, lighting, and angles).  
* **Tools:** All images were manually labeled using **Label Studio**.

### **2\. Training**

The model was trained using a transfer learning approach on a pre-trained yolo11s model.

* **Script:** The training process is automated in train.py. This script handles training, validation, and automatically organizes the final model and analytics.  
* **Analytics:** All training graphs and analytics are available in the /training\_analytics/ folder.

### **3\. Detection**

The detect.py script is the final application. It loads the custom-trained best.pt file and runs inference on a video source using OpenCV.

---

## **How to Run**

### **1\. Prerequisites**

* Python 3.8+  
* An NVIDIA GPU (for decent performance)

### **2\. Clone Repository**

### **3\. Install Dependencies**

All required libraries are in requirements.txt.

Bash

pip install \-r requirements.txt

### **4\. Run Detection**

The script detect.py is the main entry point.

**To use your webcam:**

Bash

python detect.py \--model models/best.pt \--source 0 \--resolution 1280x720

**To use a video file:**

Bash

python detect.py \--model models/best.pt \--source path/to/your\_video.mp4

---

## **Project Structure**

/football-kit-detector  
├── dataset/            \# Data config and (split) image/label files  
├── models/             \# Contains the final trained best.pt  
├── training\_analytics/ \# All output graphs (results.png, etc.)  
│  
├── prepare\_dataset.py  \# Script to auto-split raw data  
├── train.py            \# Script to run training  
├── detect.py           \# Script to run live detection (the application)  
│  
├── README.md           \# You are here\!  
├── requirements.txt  
└── .gitignore
