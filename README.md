# NAO PPE Detection

This repository contains a Python program for real-time detection of Personal Protective Equipment (PPE) using a NAO humanoid robot and a YOLO-based object detection model.

The program captures images from the NAO robot's camera, performs PPE detection using a trained YOLO model, and displays the results with bounding boxes. Optionally, the robot can announce the detected PPEs using text-to-speech.

## Repository

[https://github.com/vitor-souza-ime/naoppe](https://github.com/vitor-souza-ime/naoppe)

## File

- `main.py` — Main Python script for PPE detection with NAO.

## Requirements

- Python 3.8 or higher
- NAOqi SDK (`qi` library)
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Ultralytics YOLO (`ultralytics`)
- PyTorch
- NAO robot connected to the same network

Install dependencies:

```bash
pip install opencv-python numpy torch ultralytics
````

## Setup

1. Connect your NAO robot to the same network as your computer.
2. Update the IP address and port of your NAO robot in `main.py`:

```python
NAO_IP = "YOUR_NAO_IP"
NAO_PORT = 9559
```

3. Ensure you have your trained YOLO model (`best.pt`) and update the path in `main.py`:

```python
model_path = "/path/to/your/best.pt"
```

## Usage

Run the main script:

```bash
python main.py
```

The program will:

1. Connect to the NAO robot.
2. Capture video frames from the NAO camera.
3. Perform PPE detection using the YOLO model every 2 seconds.
4. Display detected PPEs on the screen with bounding boxes and detection info.
5. Optionally, announce detected PPEs using NAO's text-to-speech service.

Press **`q`** to exit the program.

## Notes

* The script sets the robot’s body stiffness to 0 (`motion_service.setStiffnesses("Body", 0.0)`), allowing free movement of limbs.
* Detection confidence threshold is set to 0.3.
* Detection interval is set to 2 seconds by default.

## License

This project is licensed under the MIT License.

