# SlamGeoHot

SlamGeoHot is a toy implementation of SLAM (Simultaneous Localization and Mapping), created during the Anantya Hackathon and guided by George Hotz. This project utilizes various technologies to visualize and understand the process of feature extraction and 3D representation from 2D images. The primary focus is to experiment and learn about the core aspects of SLAM technology.

## Features

- **2D Display using SDL2**: Utilizes SDL2 for rendering 2D views of the environment.
- **Feature Extraction with OpenCV**: Leverages OpenCV for detecting and extracting key points from images.
- **3D Visualization**: Employs Pangolin for dynamic 3D visualization of the mapping process.
- **Graph Optimization**: Integration of g2o for optimizing the SLAM graph (planned feature).

## Installation

### Prerequisites

Ensure you have Python 3.x, OpenCV, SDL2, and Pangolin installed on your system. You may need to build Pangolin from source, as described in its documentation.

### Setup

1. Clone the repository:

   ```
   git clone https://github.com/rohanawhad/SlamGeoHot.git
   ```

2. Navigate to the cloned directory:

   ```
   cd SlamGeoHot
   ```

3. Install required Python packages:

   ```
   pip install numpy opencv-python-headless
   ```

   Note: `opencv-python-headless` is used to avoid unnecessary GUI dependencies for OpenCV when running on headless servers.

## Usage

Run the SLAM application with:

```
python slam.py
```

The application expects a video input. To modify the video source:

- **To use a different video file**: Replace `'test_countryroad.mp4'` with the path to your new video file in the `slam.py` script:
  ```python
  cap = cv2.VideoCapture('path/to/your/video.mp4')
  ```
- **To use a webcam**: Change the source to use the default webcam:
  ```python
  cap = cv2.VideoCapture(0)
  ```

## How It Works

- **Initialization**: The system initializes by setting up the camera parameters and creating a map to store and manage 3D points and camera frames.
- **Frame Processing**: Each frame from the video is processed to detect features and establish correspondences between consecutive frames.
- **Pose Estimation**: Calculates camera pose by extracting the relative motion between frames.
- **Triangulation**: Generates 3D points from the corresponding features across frames.
- **Optimization**: Adjusts the map points and frame poses to better fit the observed data (coming soon).

## Contributing

Feel free to fork the project and submit pull requests. You can also open issues for any bugs found or features suggested.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
