"""
Disparity computation module for stereo vision depth estimation.
Implements various stereo matching algorithms.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class DisparityComputer:
    def __init__(self, method: str = "sgbm"):
        """
        Initialize disparity computer.
        
        Args:
            method: Stereo matching method ('bm' or 'sgbm')
        """
        self.method = method.lower()
        self.stereo_matcher = None
        self._initialize_matcher()
    
    def _initialize_matcher(self):
        """Initialize the stereo matcher based on selected method."""
        if self.method == "bm":
            self.stereo_matcher = cv2.StereoBM_create(
                numDisparities=16 * 6,  # Must be divisible by 16
                blockSize=15
            )
            
            # Set additional parameters for StereoBM
            self.stereo_matcher.setROI1((0, 0, 0, 0))
            self.stereo_matcher.setROI2((0, 0, 0, 0))
            self.stereo_matcher.setPreFilterCap(31)
            self.stereo_matcher.setMinDisparity(0)
            self.stereo_matcher.setTextureThreshold(10)
            self.stereo_matcher.setUniquenessRatio(15)
            self.stereo_matcher.setSpeckleWindowSize(100)
            self.stereo_matcher.setSpeckleRange(32)
            
        elif self.method == "sgbm":
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16 * 6,  # Must be divisible by 16
                blockSize=11,
                P1=8 * 3 * 11**2,  # Controls disparity smoothness
                P2=32 * 3 * 11**2,  # Controls disparity smoothness
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            raise ValueError(f"Unknown stereo method: {self.method}")
    
    def compute_disparity(self, left_frame: np.ndarray, 
                         right_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute disparity map from stereo image pair.
        
        Args:
            left_frame: Left camera frame
            right_frame: Right camera frame
            
        Returns:
            Disparity map or None if computation fails
        """
        if left_frame is None or right_frame is None:
            return None
        
        try:
            # Convert to grayscale if needed
            if len(left_frame.shape) == 3:
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_frame
                
            if len(right_frame.shape) == 3:
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            else:
                right_gray = right_frame
            
            # Compute disparity
            disparity = self.stereo_matcher.compute(left_gray, right_gray)
            
            # Convert to float and normalize
            disparity = disparity.astype(np.float32) / 16.0
            
            return disparity
            
        except Exception as e:
            print(f"Error computing disparity: {e}")
            return None
    
    def post_process_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Post-process disparity map to improve quality.
        
        Args:
            disparity: Raw disparity map
            
        Returns:
            Post-processed disparity map
        """
        # Remove invalid disparities (set them to 0)
        disparity[disparity <= 0] = 0
        
        # Apply median filter to reduce noise
        disparity_filtered = cv2.medianBlur(disparity.astype(np.uint8), 5)
        
        # Apply bilateral filter for edge-preserving smoothing
        disparity_smooth = cv2.bilateralFilter(
            disparity_filtered.astype(np.float32), 
            9, 75, 75
        )
        
        return disparity_smooth
    
    def visualize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the disparity map.
        
        Args:
            disparity: Disparity map
            
        Returns:
            Colorized disparity map for visualization
        """
        if disparity is None:
            return None
        
        # Normalize disparity for visualization
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_norm = disp_norm.astype(np.uint8)
        
        # Apply colormap
        disparity_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        
        return disparity_color
    
    def update_parameters(self, **kwargs):
        """
        Update stereo matcher parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        for param, value in kwargs.items():
            if hasattr(self.stereo_matcher, f'set{param.capitalize()}'):
                getattr(self.stereo_matcher, f'set{param.capitalize()}')(value)
            else:
                print(f"Warning: Parameter {param} not found")


def test_disparity_computation():
    """Test function for disparity computation."""
    # Create test pattern
    height, width = 480, 640
    
    # Create simple test images with horizontal shift
    left_img = np.zeros((height, width), dtype=np.uint8)
    right_img = np.zeros((height, width), dtype=np.uint8)
    
    # Add some patterns
    cv2.rectangle(left_img, (100, 100), (200, 200), 255, -1)
    cv2.rectangle(right_img, (90, 100), (190, 200), 255, -1)  # Shifted left
    
    cv2.circle(left_img, (400, 300), 50, 128, -1)
    cv2.circle(right_img, (390, 300), 50, 128, -1)  # Shifted left
    
    # Test both methods
    for method in ["bm", "sgbm"]:
        print(f"Testing {method.upper()} method...")
        
        computer = DisparityComputer(method=method)
        disparity = computer.compute_disparity(left_img, right_img)
        
        if disparity is not None:
            disparity_vis = computer.visualize_disparity(disparity)
            
            # Show results
            combined = np.hstack([
                left_img,
                right_img,
                cv2.cvtColor(disparity_vis, cv2.COLOR_BGR2GRAY)
            ])
            
            cv2.imshow(f'Test {method.upper()}: Left | Right | Disparity', combined)
            cv2.waitKey(2000)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_disparity_computation()