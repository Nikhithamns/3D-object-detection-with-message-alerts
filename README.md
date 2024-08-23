

```markdown
# Object Detection and Face Recognition with YOLO and Haar Cascade

This project implements object detection using YOLOv3 and face detection using Haar Cascade in Python. The detected objects are logged, and a WhatsApp message is sent using `pywhatkit` when new objects are detected.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Logging and WhatsApp Messaging](#logging-and-whatsapp-messaging)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/object-detection-yolo-haarcascade.git
   cd object-detection-yolo-haarcascade
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv3 files:**
   - `yolov3.cfg`: [YOLOv3 Config File](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - `yolov3.weights`: [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)
   - `coco.names`: [COCO Names File](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

   Save these files in the appropriate directory and update the paths in the configuration section of `main.py`.

## Project Structure

```
object-detection-yolo-haarcascade/
│
├── main.py                    # Main script for object detection and face recognition
├── detection_log.txt          # Log file for detected objects (auto-generated)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Usage

1. **Run the script:**

   ```bash
   python main.py
   ```

   The script captures video from the webcam (or a video file if configured), performs object detection using YOLOv3, and detects faces using Haar Cascade. Detected objects are logged, and a WhatsApp message is sent if new objects are identified.

2. **Terminate the script:**
   - Press `q` to quit the video feed and stop the detection process.

## Configuration

Before running the script, make sure to configure the following paths in `main.py`:

```python
FACE_CASCADE_PATH = '/path/to/haarcascade_frontalface_default.xml'
YOLO_CONFIG_FILE = '/path/to/yolov3.cfg'
YOLO_WEIGHTS_FILE = '/path/to/yolov3.weights'
NAMES_FILE = '/path/to/coco.names'
WHATSAPP_PHONE_NUMBER = '+91XXXXXXXXXX'  # Your phone number for WhatsApp messaging
```

## Dependencies

The project requires the following Python packages:

- `opencv-python`
- `numpy`
- `pywhatkit`

Install them using:

```bash
pip install -r requirements.txt
```

## Logging and WhatsApp Messaging

- **Logging**: Detected objects are logged in `detection_log.txt` with a timestamp and confidence score.
- **WhatsApp Messaging**: When a new object is detected, a WhatsApp message is sent to the configured phone number using `pywhatkit`.

## Troubleshooting

### Common Issues

1. **Haar Cascade not loading**: Ensure the correct path to the Haar Cascade XML file and verify the file's existence.
2. **YOLO files not loading**: Double-check the paths to the YOLO configuration and weights files.
3. **OpenCV errors**: If you encounter errors related to OpenCV, try reinstalling it with:
   ```bash
   pip uninstall opencv-python opencv-python-headless
   pip install opencv-python
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```

### Instructions
1. **Edit the paths**: Ensure that the paths provided in the `Configuration` section match your local setup.
2. **Dependencies**: Make sure to include the `requirements.txt` in your project directory with the necessary dependencies listed.
