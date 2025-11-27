# Collision Warning System

A real-time collision detection and warning application using YOLOv8 object detection and Time-To-Collision (TTC) calculation to prevent vehicle collisions.

## Overview

This application monitors video feeds to detect vehicles and pedestrians, calculates the time remaining until a collision occurs, and alerts the user through visual warnings and audio alerts when a collision risk is detected.

### Key Features

- **Real-time Object Detection**: Uses YOLOv8 for detecting vehicles (cars, motorcycles, buses, trucks) and pedestrians
- **Time-To-Collision (TTC) Calculation**: Calculates the time available before a potential collision
- **Multi-object Tracking**: Tracks multiple objects across video frames
- **Visual Alerts**: Real-time HUD display showing collision risk level (Safe, Warning, Danger)
- **Audio Alerts**: Configurable sound notifications when collision risk is detected
- **Desktop GUI**: User-friendly Tkinter interface with video playback controls
- **Configurable Settings**: Adjust TTC thresholds, AI models, and alert preferences
- **Video File Support**: Load and analyze custom video files

## Project Structure

```
collision_warning_system/
├── desktop_app.py              # Main desktop application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── backend/
│   ├── src/
│   │   ├── app.py             # FastAPI backend server
│   │   └── processing/
│   │       ├── detector.py    # YOLOv8 object detection module
│   │       └── calculator.py  # TTC calculation module
│   └── requirements.txt        # Backend dependencies
├── ui/
│   ├── main_window_ui.py      # Main window UI
│   ├── settings_window_ui.py  # Settings dialog UI
│   └── about_window_ui.py     # About dialog UI
├── data/
│   └── videos/                # Sample video files for testing
├── models/
│   └── yolov8n.pt            # YOLOv8 nano model weights
└── assets/
    └── sounds/                # Alert sound files
```

## Requirements

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/haooah01/Group1-Application-Detect-And-Report-To-Prevent-Collisions.git
cd collision_warning_system
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Desktop Application

```bash
python desktop_app.py
```

This launches the main GUI with:
- Video playback controls (Play/Pause/Stop)
- Real-time collision detection visualization
- Settings and About windows
- Live FPS and object count display

### Running the Backend API Server

```bash
python backend/src/app.py
```

The API server runs on `http://127.0.0.1:8000` and provides:
- `/video_stream` - Live video stream with detections
- `/` - Web interface for monitoring

## How It Works

### 1. Object Detection
The system uses YOLOv8 (nano model) to detect:
- Pedestrians (Class 1)
- Cars (Class 2)
- Motorcycles (Class 3)
- Buses (Class 5)
- Trucks (Class 7)

### 2. Tracking
Objects are tracked across frames using YOLO's built-in tracking system.

### 3. TTC Calculation
Time-To-Collision is calculated based on:
- Object bounding box position and size changes
- Object velocity estimation
- Distance to collision point
- Frame rate

### 4. Alert System
- **Green (Safe)**: TTC > 3.0 seconds
- **Yellow (Warning)**: 2.0 < TTC ≤ 3.0 seconds
- **Red (Danger)**: TTC ≤ 2.0 seconds - Audio alert plays continuously

## Configuration

Access settings through the application's Settings window to configure:

- **TTC Threshold**: Adjust collision warning threshold (default: 3.0s)
- **AI Model**: Switch between different YOLO models
- **Sound Alerts**: Enable/disable audio notifications

## Technical Details

### TTC Calculation Algorithm
```
TTC = Distance / Velocity
```
Where:
- **Distance**: Calculated from bounding box dimensions
- **Velocity**: Estimated from frame-to-frame position changes

### Region of Interest (ROI)
The system focuses on the center 60% of the video frame to concentrate on vehicles directly ahead.

## Performance

- **Model**: YOLOv8n (nano - lightweight, fast)
- **Target FPS**: Real-time processing (30+ FPS on modern hardware)
- **Memory**: Optimized for both CPU and GPU execution

## File Descriptions

| File | Purpose |
|------|---------|
| `desktop_app.py` | Main Tkinter application with UI and logic integration |
| `backend/src/app.py` | FastAPI server for web-based monitoring |
| `backend/src/processing/detector.py` | YOLOv8 detection and tracking |
| `backend/src/processing/calculator.py` | TTC computation and collision warning logic |
| `ui/*.py` | UI components (main window, settings, about) |

## Dependencies

- **ultralytics** - YOLOv8 framework
- **opencv-python** - Video processing and visualization
- **Pillow** - Image processing
- **pygame** - Audio playback
- **fastapi** - Web API framework (optional, for server)
- **uvicorn** - ASGI web server (optional, for server)

## Future Enhancements

- [ ] Support for live camera feeds (webcam)
- [ ] Multi-camera support
- [ ] Data logging and analysis
- [ ] Mobile application
- [ ] Advanced ML models for improved accuracy
- [ ] Dash cam integration
- [ ] Cloud-based monitoring dashboard

## License

This project is developed by Group 1 for vehicle collision detection and prevention.

## Contact

For questions or contributions, please visit the [GitHub Repository](https://github.com/haooah01/Group1-Application-Detect-And-Report-To-Prevent-Collisions)

## Disclaimer

This system is designed as a safety assistance tool. It should not be relied upon as the sole means of collision prevention. Always maintain awareness of your surroundings while driving.
