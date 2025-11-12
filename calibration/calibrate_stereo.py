"""
Basic stereo camera calibration using chessboard pattern.
"""

import cv2
import numpy as np
import os
import glob
import yaml
from typing import Tuple, List, Optional

def create_chessboard_corners(pattern_size: Tuple[int, int], square_size: float = 1.0) -> np.ndarray:
    """
    Create 3D coordinates for chessboard corners.
    
    Args:
        pattern_size: (width, height) number of inner corners
        square_size: Size of each square in real units
        
    Returns:
        3D coordinates array
    """
    corners_3d = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    corners_3d[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    corners_3d *= square_size
    return corners_3d

def find_chessboard_corners(image: np.ndarray, pattern_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Find chessboard corners in an image.
    
    Args:
        image: Input image
        pattern_size: (width, height) number of inner corners
        
    Returns:
        Corner coordinates or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners
    
    return None

def calibrate_single_camera(images: List[np.ndarray], pattern_size: Tuple[int, int], 
                           square_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Calibrate a single camera.
    
    Args:
        images: List of calibration images
        pattern_size: Chessboard pattern size
        square_size: Size of squares
        
    Returns:
        Tuple of (camera_matrix, dist_coeffs, rvecs, tvecs)
    """
    # Prepare object points
    objp = create_chessboard_corners(pattern_size, square_size)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    image_size = None
    
    for img in images:
        corners = find_chessboard_corners(img, pattern_size)
        
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            if image_size is None:
                image_size = img.shape[:2][::-1]  # (width, height)
    
    if len(objpoints) < 3:
        raise ValueError("Not enough valid calibration images found")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    return camera_matrix, dist_coeffs, rvecs, tvecs

def calibrate_stereo_cameras(left_images: List[np.ndarray], right_images: List[np.ndarray],
                           pattern_size: Tuple[int, int], square_size: float = 1.0) -> dict:
    """
    Calibrate stereo camera system.
    
    Args:
        left_images: Left camera images
        right_images: Right camera images  
        pattern_size: Chessboard pattern size
        square_size: Size of squares
        
    Returns:
        Dictionary with all calibration parameters
    """
    print(f"Calibrating with {len(left_images)} image pairs...")
    
    # Prepare object points
    objp = create_chessboard_corners(pattern_size, square_size)
    
    # Arrays to store points
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    
    image_size = None
    
    # Find corners in both images
    valid_pairs = 0
    for left_img, right_img in zip(left_images, right_images):
        corners_left = find_chessboard_corners(left_img, pattern_size)
        corners_right = find_chessboard_corners(right_img, pattern_size)
        
        if corners_left is not None and corners_right is not None:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            valid_pairs += 1
            
            if image_size is None:
                image_size = left_img.shape[:2][::-1]
    
    print(f"Found {valid_pairs} valid image pairs")
    
    if valid_pairs < 5:
        raise ValueError("Not enough valid image pairs for stereo calibration")
    
    # Calibrate individual cameras first
    print("Calibrating left camera...")
    camera_matrix_left, dist_coeffs_left, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None
    )
    
    print("Calibrating right camera...")
    camera_matrix_right, dist_coeffs_right, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None
    )
    
    # Stereo calibration
    print("Performing stereo calibration...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, \
    R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    print(f"Stereo calibration RMS error: {ret:.3f}")
    
    return {
        'camera_matrix_left': camera_matrix_left,
        'dist_coeffs_left': dist_coeffs_left,
        'camera_matrix_right': camera_matrix_right,
        'dist_coeffs_right': dist_coeffs_right,
        'rotation_matrix': R,
        'translation_vector': T,
        'essential_matrix': E,
        'fundamental_matrix': F,
        'rms_error': ret,
        'image_size': image_size
    }

def save_calibration_data(calibration_data: dict, output_dir: str):
    """Save calibration data to YAML files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save left camera parameters
    left_data = {
        'camera_matrix': calibration_data['camera_matrix_left'].tolist(),
        'dist_coeffs': calibration_data['dist_coeffs_left'].tolist()
    }
    with open(os.path.join(output_dir, 'left_cam.yaml'), 'w') as f:
        yaml.dump(left_data, f)
    
    # Save right camera parameters
    right_data = {
        'camera_matrix': calibration_data['camera_matrix_right'].tolist(),
        'dist_coeffs': calibration_data['dist_coeffs_right'].tolist()
    }
    with open(os.path.join(output_dir, 'right_cam.yaml'), 'w') as f:
        yaml.dump(right_data, f)
    
    # Save stereo parameters
    stereo_data = {
        'rotation_matrix': calibration_data['rotation_matrix'].tolist(),
        'translation_vector': calibration_data['translation_vector'].tolist(),
        'essential_matrix': calibration_data['essential_matrix'].tolist(),
        'fundamental_matrix': calibration_data['fundamental_matrix'].tolist(),
        'rms_error': float(calibration_data['rms_error']),
        'image_size': calibration_data['image_size']
    }
    with open(os.path.join(output_dir, 'stereo_params.yaml'), 'w') as f:
        yaml.dump(stereo_data, f)
    
    print(f"Calibration data saved to {output_dir}")

def load_images_from_directory(directory: str) -> List[np.ndarray]:
    """Load all images from a directory."""
    images = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    for ext in extensions:
        files = glob.glob(os.path.join(directory, ext))
        files.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        for file in sorted(files):
            img = cv2.imread(file)
            if img is not None:
                images.append(img)
    
    return images

def main():
    """Main calibration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stereo camera calibration")
    parser.add_argument("--left-dir", required=True, help="Directory with left camera images")
    parser.add_argument("--right-dir", required=True, help="Directory with right camera images")
    parser.add_argument("--output-dir", default="calibration/camera_param", 
                       help="Output directory for calibration files")
    parser.add_argument("--pattern-width", type=int, default=9, 
                       help="Number of inner corners in width")
    parser.add_argument("--pattern-height", type=int, default=6,
                       help="Number of inner corners in height")
    parser.add_argument("--square-size", type=float, default=1.0,
                       help="Size of each square in real units")
    
    args = parser.parse_args()
    
    # Load images
    print(f"Loading images from {args.left_dir} and {args.right_dir}")
    left_images = load_images_from_directory(args.left_dir)
    right_images = load_images_from_directory(args.right_dir)
    
    print(f"Loaded {len(left_images)} left images and {len(right_images)} right images")
    
    if len(left_images) != len(right_images):
        print("Warning: Different number of left and right images")
        min_count = min(len(left_images), len(right_images))
        left_images = left_images[:min_count]
        right_images = right_images[:min_count]
    
    if len(left_images) == 0:
        print("No images found!")
        return
    
    try:
        # Perform calibration
        pattern_size = (args.pattern_width, args.pattern_height)
        calibration_data = calibrate_stereo_cameras(
            left_images, right_images, pattern_size, args.square_size
        )
        
        # Save results
        save_calibration_data(calibration_data, args.output_dir)
        
        print("Calibration completed successfully!")
        print(f"RMS error: {calibration_data['rms_error']:.3f}")
        
        # Print baseline distance
        baseline = np.linalg.norm(calibration_data['translation_vector'])
        print(f"Baseline distance: {baseline:.2f} units")
        
    except Exception as e:
        print(f"Calibration failed: {e}")

if __name__ == "__main__":
    main()

