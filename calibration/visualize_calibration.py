"""
Visualize stereo calibration results and test rectification.
"""

import cv2
import numpy as np
import yaml
import os
import glob
from typing import Tuple, Optional

def load_calibration_data(calibration_dir: str) -> dict:
    """Load calibration data from YAML files."""
    calibration_data = {}
    
    # Load left camera
    left_file = os.path.join(calibration_dir, "left_cam.yaml")
    if os.path.exists(left_file):
        with open(left_file, 'r') as f:
            left_data = yaml.safe_load(f)
            calibration_data['camera_matrix_left'] = np.array(left_data['camera_matrix'])
            calibration_data['dist_coeffs_left'] = np.array(left_data['dist_coeffs'])
    
    # Load right camera
    right_file = os.path.join(calibration_dir, "right_cam.yaml")
    if os.path.exists(right_file):
        with open(right_file, 'r') as f:
            right_data = yaml.safe_load(f)
            calibration_data['camera_matrix_right'] = np.array(right_data['camera_matrix'])
            calibration_data['dist_coeffs_right'] = np.array(right_data['dist_coeffs'])
    
    # Load stereo parameters
    stereo_file = os.path.join(calibration_dir, "stereo_params.yaml")
    if os.path.exists(stereo_file):
        with open(stereo_file, 'r') as f:
            stereo_data = yaml.safe_load(f)
            calibration_data['rotation_matrix'] = np.array(stereo_data['rotation_matrix'])
            calibration_data['translation_vector'] = np.array(stereo_data['translation_vector'])
            calibration_data['image_size'] = tuple(stereo_data['image_size'])
    
    return calibration_data

def compute_rectification_maps(calibration_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification maps."""
    # Extract calibration parameters
    camera_matrix_left = calibration_data['camera_matrix_left']
    dist_coeffs_left = calibration_data['dist_coeffs_left']
    camera_matrix_right = calibration_data['camera_matrix_right']
    dist_coeffs_right = calibration_data['dist_coeffs_right']
    R = calibration_data['rotation_matrix']
    T = calibration_data['translation_vector']
    image_size = calibration_data['image_size']
    
    # Compute rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1
    )
    
    # Compute rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        camera_matrix_left, dist_coeffs_left, R1, P1, image_size, cv2.CV_16SC2
    )
    
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        camera_matrix_right, dist_coeffs_right, R2, P2, image_size, cv2.CV_16SC2
    )
    
    return map1_left, map2_left, map1_right, map2_right, Q

def rectify_image_pair(left_img: np.ndarray, right_img: np.ndarray,
                      map1_left: np.ndarray, map2_left: np.ndarray,
                      map1_right: np.ndarray, map2_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rectify a pair of stereo images."""
    left_rectified = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
    return left_rectified, right_rectified

def draw_epipolar_lines(img1: np.ndarray, img2: np.ndarray, step: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Draw epipolar lines on rectified images."""
    h, w = img1.shape[:2]
    
    # Draw horizontal lines
    for y in range(0, h, step):
        cv2.line(img1, (0, y), (w-1, y), (0, 255, 0), 1)
        cv2.line(img2, (0, y), (w-1, y), (0, 255, 0), 1)
    
    return img1, img2

def visualize_rectification(left_images_dir: str, right_images_dir: str, calibration_dir: str):
    """Visualize rectification results."""
    # Load calibration data
    print(f"Loading calibration from {calibration_dir}")
    calibration_data = load_calibration_data(calibration_dir)
    
    if not calibration_data:
        print("No calibration data found!")
        return
    
    # Compute rectification maps
    print("Computing rectification maps...")
    map1_left, map2_left, map1_right, map2_right, Q = compute_rectification_maps(calibration_data)
    
    # Load test images
    left_files = sorted(glob.glob(os.path.join(left_images_dir, "*.jpg")) + 
                       glob.glob(os.path.join(left_images_dir, "*.png")))
    right_files = sorted(glob.glob(os.path.join(right_images_dir, "*.jpg")) + 
                        glob.glob(os.path.join(right_images_dir, "*.png")))
    
    if not left_files or not right_files:
        print("No test images found!")
        return
    
    print(f"Found {len(left_files)} left images and {len(right_files)} right images")
    print("Press 'n' for next image pair, 'p' for previous, 'q' to quit")
    
    image_index = 0
    
    while True:
        if image_index >= len(left_files) or image_index >= len(right_files):
            image_index = 0
        
        # Load current image pair
        left_img = cv2.imread(left_files[image_index])
        right_img = cv2.imread(right_files[image_index])
        
        if left_img is None or right_img is None:
            print(f"Failed to load images at index {image_index}")
            image_index += 1
            continue
        
        # Rectify images
        left_rect, right_rect = rectify_image_pair(
            left_img, right_img, map1_left, map2_left, map1_right, map2_right
        )
        
        # Draw epipolar lines
        left_lines, right_lines = draw_epipolar_lines(left_rect.copy(), right_rect.copy())
        
        # Create comparison views
        # Original images side by side
        original_combined = np.hstack([
            cv2.resize(left_img, (400, 300)),
            cv2.resize(right_img, (400, 300))
        ])
        
        # Rectified images with epipolar lines
        rectified_combined = np.hstack([
            cv2.resize(left_lines, (400, 300)),
            cv2.resize(right_lines, (400, 300))
        ])
        
        # Combine top and bottom
        display = np.vstack([original_combined, rectified_combined])
        
        # Add text labels
        cv2.putText(display, "Original Images", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Rectified with Epipolar Lines", (10, 330), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Image {image_index + 1}/{len(left_files)}", (10, 580), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Calibration Visualization', display)
        
        # Handle key presses
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            image_index += 1
        elif key == ord('p'):
            image_index = max(0, image_index - 1)
    
    cv2.destroyAllWindows()
    print("Visualization completed")

def print_calibration_info(calibration_dir: str):
    """Print calibration information."""
    calibration_data = load_calibration_data(calibration_dir)
    
    if not calibration_data:
        print("No calibration data found!")
        return
    
    print("=== Stereo Calibration Information ===")
    
    if 'camera_matrix_left' in calibration_data:
        K_left = calibration_data['camera_matrix_left']
        print(f"\nLeft Camera Matrix:")
        print(f"  fx = {K_left[0,0]:.2f}, fy = {K_left[1,1]:.2f}")
        print(f"  cx = {K_left[0,2]:.2f}, cy = {K_left[1,2]:.2f}")
        
        dist_left = calibration_data['dist_coeffs_left']
        print(f"  Distortion: {dist_left.flatten()}")
    
    if 'camera_matrix_right' in calibration_data:
        K_right = calibration_data['camera_matrix_right']
        print(f"\nRight Camera Matrix:")
        print(f"  fx = {K_right[0,0]:.2f}, fy = {K_right[1,1]:.2f}")
        print(f"  cx = {K_right[0,2]:.2f}, cy = {K_right[1,2]:.2f}")
        
        dist_right = calibration_data['dist_coeffs_right']
        print(f"  Distortion: {dist_right.flatten()}")
    
    if 'translation_vector' in calibration_data:
        T = calibration_data['translation_vector']
        baseline = np.linalg.norm(T)
        print(f"\nStereo Parameters:")
        print(f"  Baseline: {baseline:.2f} units")
        print(f"  Translation: [{T[0]:.3f}, {T[1]:.3f}, {T[2]:.3f}]")
        
        if 'rotation_matrix' in calibration_data:
            R = calibration_data['rotation_matrix']
            # Convert rotation matrix to Euler angles
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(R[2,1], R[2,2])
                y = np.arctan2(-R[2,0], sy)
                z = np.arctan2(R[1,0], R[0,0])
            else:
                x = np.arctan2(-R[1,2], R[1,1])
                y = np.arctan2(-R[2,0], sy)
                z = 0
            
            print(f"  Rotation (deg): [{np.degrees(x):.2f}, {np.degrees(y):.2f}, {np.degrees(z):.2f}]")

def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize stereo calibration")
    parser.add_argument("--calibration-dir", default="calibration/camera_param",
                       help="Directory containing calibration files")
    parser.add_argument("--left-images", default="calibration/chessboard_images_left",
                       help="Directory with left test images")
    parser.add_argument("--right-images", default="calibration/chessboard_images_right", 
                       help="Directory with right test images")
    parser.add_argument("--info-only", action="store_true",
                       help="Only print calibration info, don't show visualization")
    
    args = parser.parse_args()
    
    if args.info_only:
        print_calibration_info(args.calibration_dir)
    else:
        print_calibration_info(args.calibration_dir)
        visualize_rectification(args.left_images, args.right_images, args.calibration_dir)

if __name__ == "__main__":
    main()