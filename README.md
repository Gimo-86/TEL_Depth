# Real-Time Depth Estimation from Dual Webcams

A comprehensive stereo vision system that computes real-time depth maps using two webcams. The system captures synchronized frames, computes disparity using stereo matching algorithms, and converts disparity to depth measurements.

## Features

- **Real-time stereo capture** from dual webcams
- **Multiple stereo matching algorithms** (Block Matching, Semi-Global Block Matching)
- **Interactive depth visualization** with measurement tools
- **Stereo camera calibration** using chessboard patterns
- **Multiple display modes** (quad view, depth only, stereo, anaglyph)
- **Performance monitoring** with FPS and processing time metrics
- **Data recording** for offline analysis
- **Point cloud generation** from depth maps

## Quick Start

### Prerequisites

- Python 3.7+
- Two webcams connected to your system
- OpenCV, NumPy, PyYAML, matplotlib (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TEL_Depth
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run a quick demo (10 seconds):
```bash
python demo.py
```

### Basic Usage

1. **Quick test** with default settings:
```bash
python demo.py --duration 30
```

2. **Full application** with all features:
```bash
python main.py
```

3. **Custom camera setup**:
```bash
python main.py --left-camera 0 --right-camera 2 --baseline 80
```

## Controls

### During Operation

- **ESC**: Quit application
- **Q/1**: Quad view (left, right, depth, disparity)
- **D/2**: Depth map only
- **S/3**: Stereo view (left and right side by side)
- **A/4**: Red-cyan anaglyph view
- **I**: Toggle information overlay
- **R**: Toggle recording
- **P**: Toggle post-processing
- **T**: Toggle rectification
- **C**: Clear depth measurement point
- **+/-**: Adjust depth display range

### Mouse Controls

- **Left click**: Measure depth at clicked point
- Works in all view modes with automatic coordinate adjustment

## Camera Setup

### Hardware Requirements

1. **Two identical webcams** (recommended)
2. **Fixed mounting** to prevent movement during operation
3. **Baseline distance** of 50-150mm for best results
4. **Good lighting** for optimal stereo matching

### Camera Positioning

```
Left Camera    Right Camera
    |              |
    |<--baseline-->|
    |              |
    v              v
   Scene to be measured
```

- Mount cameras horizontally aligned
- Ensure both cameras point forward
- Minimize vertical misalignment
- Typical baseline: 60-100mm for desktop use

## Calibration

For best results, calibrate your stereo camera setup:

### 1. Collect Calibration Images

```bash
# Create directories for calibration images
mkdir -p calibration/chessboard_images_left
mkdir -p calibration/chessboard_images_right
```

Capture 15-30 image pairs of a chessboard pattern:
- Print a chessboard pattern (9x6 inner corners recommended)
- Capture images with the pattern at various positions and orientations
- Ensure both cameras see the complete pattern in each image
- Save left camera images to `calibration/chessboard_images_left/`
- Save right camera images to `calibration/chessboard_images_right/`

### 2. Run Calibration

```bash
python calibration/calibrate_stereo.py \
    --left-dir calibration/chessboard_images_left \
    --right-dir calibration/chessboard_images_right \
    --pattern-width 9 --pattern-height 6
```

### 3. Verify Calibration

```bash
python calibration/visualize_calibration.py
```

This shows epipolar lines on rectified images to verify calibration quality.

## Advanced Usage

### Command Line Options

```bash
python main.py --help
```

Key parameters:
- `--focal-length`: Camera focal length in pixels (default: 700)
- `--baseline`: Distance between cameras in mm (default: 60)
- `--stereo-method`: Matching algorithm (bm/sgbm, default: sgbm)
- `--max-frames`: Limit processing to N frames

### Recording Data

Enable recording to save:
- Raw stereo image pairs
- Depth maps (as numpy arrays)
- Disparity maps
- Visualization images

Press 'R' during operation or use `--record` flag.

### Output Structure

```
output/
├── frames/           # Raw camera images
├── depth_maps/       # Depth data (.npy) and visualizations  
├── disparity_maps/   # Disparity data and visualizations
├── point_clouds/     # 3D point cloud data
└── rectified/        # Rectified stereo pairs
```

## Performance Tuning

### For Better Speed
- Use Block Matching (`--stereo-method bm`)
- Reduce image resolution in camera settings
- Disable post-processing (press 'P')
- Use lower-resolution cameras

### For Better Quality
- Use SGBM algorithm (default)
- Proper camera calibration
- Good lighting conditions
- Enable post-processing
- Higher resolution cameras

### Typical Performance
- 640x480 resolution: 15-25 FPS
- 1280x720 resolution: 5-10 FPS
- Processing time: 20-60ms per frame

## Troubleshooting

### Common Issues

**"Failed to initialize cameras"**
- Check camera connections
- Try different camera IDs (0,1 or 0,2 etc.)
- Ensure cameras aren't used by other applications
- Check camera permissions

**Poor depth quality**
- Improve lighting
- Calibrate cameras properly
- Check camera alignment
- Adjust baseline distance
- Clean camera lenses

**Low FPS**
- Use faster stereo method (BM vs SGBM)
- Reduce image resolution
- Close other applications
- Use USB 3.0 cameras

**Depth measurements seem wrong**
- Verify baseline parameter matches physical setup
- Check focal length parameter
- Perform proper calibration
- Ensure cameras are properly aligned

### Parameter Tuning

Key parameters to adjust:
- **Focal length**: Affects depth scale (measure or estimate)
- **Baseline**: Physical distance between cameras
- **Disparity range**: Adjust based on scene depth
- **Block size**: Larger for smoother results, smaller for detail

## File Structure

```
TEL_Depth/
├── main.py                 # Main application
├── demo.py                 # Quick demo script  
├── requirements.txt        # Python dependencies
├── src/
│   ├── capture_stereo.py   # Camera capture
│   ├── compute_disparity.py # Stereo matching
│   ├── depth_estimation.py # Depth conversion
│   ├── visualize_depth.py  # Display and UI
│   └── utils.py           # Utility functions
├── calibration/
│   ├── calibrate_stereo.py    # Calibration script
│   ├── visualize_calibration.py # Calibration verification
│   └── camera_param/          # Calibration results
└── output/                    # Generated data
```

## Technical Details

### Depth Calculation

Depth is computed using the standard stereo vision formula:
```
depth = (focal_length × baseline) / disparity
```

Where:
- `focal_length`: Camera focal length in pixels
- `baseline`: Physical distance between cameras
- `disparity`: Pixel difference between corresponding points

### Coordinate Systems

- **Image coordinates**: (0,0) at top-left
- **Depth units**: Millimeters (configurable)
- **3D coordinates**: Camera-centered coordinate system

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## License

This project is released under the MIT License. See LICENSE file for details.

## References

- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html)
- [Stereo Vision and Depth Estimation](https://web.stanford.edu/class/cs231a/course_notes/04-stereo.pdf)
