"""
Main depth estimation module that converts disparity to depth.
Handles calibration and 3D reconstruction.
"""

import cv2
import numpy as np
import yaml
import os
from typing import Optional, Tuple, Dict


class DepthEstimator:
    def __init__(self, focal_length: float = 700.0, baseline: float = 60.0):
        """
        Initialize depth estimator.
        
        Args:
            focal_length: Camera focal length in pixels
            baseline: Distance between cameras in mm
        """
        self.focal_length = focal_length
        self.baseline = baseline
        
        # Camera calibration matrices
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.rotation_matrix = None
        self.translation_vector = None
        
        # Stereo rectification maps
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None
        
        # Q matrix for 3D reconstruction
        self.Q = None
        
    def load_calibration(self, calibration_dir: str) -> bool:
        """
        Load camera calibration parameters from YAML files.
        
        Args:
            calibration_dir: Directory containing calibration files
            
        Returns:
            True if calibration loaded successfully
        """
        try:
            left_cam_file = os.path.join(calibration_dir, "left_cam.yaml")
            right_cam_file = os.path.join(calibration_dir, "right_cam.yaml")
            stereo_file = os.path.join(calibration_dir, "stereo_params.yaml")
            
            # Load left camera parameters
            if os.path.exists(left_cam_file):
                with open(left_cam_file, 'r') as f:
                    left_data = yaml.safe_load(f)
                    self.camera_matrix_left = np.array(left_data['camera_matrix'])
                    self.dist_coeffs_left = np.array(left_data['dist_coeffs'])
            
            # Load right camera parameters
            if os.path.exists(right_cam_file):
                with open(right_cam_file, 'r') as f:
                    right_data = yaml.safe_load(f)
                    self.camera_matrix_right = np.array(right_data['camera_matrix'])
                    self.dist_coeffs_right = np.array(right_data['dist_coeffs'])
            
            # Load stereo parameters
            if os.path.exists(stereo_file):
                with open(stereo_file, 'r') as f:
                    stereo_data = yaml.safe_load(f)
                    self.rotation_matrix = np.array(stereo_data['rotation_matrix'])
                    self.translation_vector = np.array(stereo_data['translation_vector'])
                    self.baseline = np.linalg.norm(self.translation_vector)
            
            print("Calibration loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            print("Using default parameters")
            return False
    
    def setup_default_calibration(self, image_size: Tuple[int, int]):
        """
        Set up default calibration parameters.
        
        Args:
            image_size: (width, height) of the images
        """
        width, height = image_size
        
        # Default camera matrix (assuming similar cameras)
        self.camera_matrix_left = np.array([
            [self.focal_length, 0, width/2],
            [0, self.focal_length, height/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.camera_matrix_right = self.camera_matrix_left.copy()
        
        # Assume no distortion
        self.dist_coeffs_left = np.zeros(5, dtype=np.float32)
        self.dist_coeffs_right = np.zeros(5, dtype=np.float32)
        
        # Default stereo parameters (purely horizontal baseline)
        self.rotation_matrix = np.eye(3, dtype=np.float32)
        self.translation_vector = np.array([self.baseline, 0, 0], dtype=np.float32)
        
    def compute_rectification_maps(self, image_size: Tuple[int, int]):
        """
        Compute stereo rectification maps.
        
        Args:
            image_size: (width, height) of the images
        """
        if (self.camera_matrix_left is None or 
            self.camera_matrix_right is None):
            return False
        
        # Compute stereo rectification
        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            image_size,
            self.rotation_matrix, self.translation_vector,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1
        )
        
        # Compute rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, R1, P1,
            image_size, cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, R2, P2,
            image_size, cv2.CV_16SC2
        )
        
        return True
    
    def rectify_images(self, left_img: np.ndarray, 
                      right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            
        Returns:
            Tuple of rectified (left, right) images
        """
        if self.map1_left is None:
            return left_img, right_img
        
        left_rect = cv2.remap(left_img, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        return left_rect, right_rect
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map.
        
        Args:
            disparity: Disparity map
            
        Returns:
            Depth map in mm
        """
        if disparity is None:
            return None
        
        # Avoid division by zero
        disparity_safe = disparity.copy()
        disparity_safe[disparity_safe <= 0] = 0.1
        
        # Compute depth using the disparity formula: depth = (focal_length * baseline) / disparity
        depth = (self.focal_length * self.baseline) / disparity_safe
        
        # Set invalid depths to 0
        depth[disparity <= 0] = 0
        
        return depth
    
    def create_point_cloud(self, disparity: np.ndarray, 
                          left_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Create 3D point cloud from disparity map.
        
        Args:
            disparity: Disparity map
            left_img: Left camera image for color information
            
        Returns:
            Point cloud as Nx6 array (X, Y, Z, R, G, B)
        """
        if disparity is None or self.Q is None:
            return None
        
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # Get valid points
        mask = disparity > 0
        points = points_3d[mask]
        
        # Get colors
        if len(left_img.shape) == 3:
            colors = left_img[mask]
        else:
            colors = np.stack([left_img[mask]] * 3, axis=-1)
        
        # Combine points and colors
        point_cloud = np.hstack([points, colors])
        
        return point_cloud
    
    def visualize_depth(self, depth: np.ndarray, max_distance: float = 2000.0) -> np.ndarray:
        """
        Create a visualization of the depth map.
        
        Args:
            depth: Depth map in mm
            max_distance: Maximum distance for visualization in mm
            
        Returns:
            Colorized depth map
        """
        if depth is None:
            return None
        
        # Normalize depth for visualization
        depth_norm = depth.copy()
        depth_norm[depth_norm > max_distance] = max_distance
        depth_norm = depth_norm / max_distance * 255
        depth_norm = depth_norm.astype(np.uint8)
        
        # Apply colormap (closer = warmer colors)
        depth_color = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_JET)
        
        # Set invalid regions to black
        depth_color[depth <= 0] = [0, 0, 0]
        
        return depth_color


def test_depth_estimation():
    """Test function for depth estimation."""
    print("Testing depth estimation with synthetic data...")
    
    # Create test disparity map
    height, width = 480, 640
    disparity = np.zeros((height, width), dtype=np.float32)
    
    # Add some test patterns with different disparities
    cv2.rectangle(disparity, (100, 100), (200, 200), 20, -1)  # Close object
    cv2.rectangle(disparity, (300, 150), (400, 250), 10, -1)  # Far object
    cv2.circle(disparity, (500, 300), 50, 15, -1)  # Medium distance
    
    # Initialize depth estimator
    estimator = DepthEstimator(focal_length=700, baseline=60)
    estimator.setup_default_calibration((width, height))
    
    # Compute depth
    depth = estimator.disparity_to_depth(disparity)
    
    # Visualize results
    depth_vis = estimator.visualize_depth(depth)
    
    if depth_vis is not None:
        cv2.imshow('Test Depth Map', depth_vis)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        print("Depth estimation test completed")
    else:
        print("Failed to create depth visualization")


if __name__ == "__main__":
    test_depth_estimation()