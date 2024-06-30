# Automatic Offside Detection System

## Project Structure

- **datasets**: This folder contains our training dataset for YOLOv8.
- **offline_images**: This folder contains images to detect whether there is an offside.
- **results**: This folder contains the inference results.
- **train.py**: This script is used for training the YOLOv8 model.
- **test.py**: This script generates object detection results from the YOLO model.
- **cluster.py**: This script is used for team classification.
- **line.py**: This script is responsible for drawing the final offside line.

## File Descriptions

### datasets
This folder includes the training dataset necessary for the YOLOv8 model. Properly labeled data helps in training an accurate object detection model.

### offside_images
In this folder, you can place the images you want to analyze for potential offsides. The system will process these images to determine if an offside situation exists.

### results
After running the inference, the output results will be stored in this folder. You can find the detection outcomes and processed images here.

### train.py
This script is essential for training the YOLOv8 model. Ensure you have your datasets ready in the `datasets` folder before running this script.

### test.py
Use this script to generate object detection results from the YOLO model. It takes the trained model and applies it to new images to detect objects.

### cluster.py
Team classification is handled by this script. It processes the detected objects and classifies them into respective teams.

### line.py
The final step of the detection system involves drawing the offside line. This script takes the classified objects and draws the offside line based on the positions of the players and the ball.

---
