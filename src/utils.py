"""
Utility functions for the depth estimation project.
"""

import cv2
import numpy as np
import os
import time
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt


def create_directories(base_path: str) -> bool:
    """
    Create necessary output directories.
    
    Args:
        base_path: Base path for the project
        
    Returns:
        True if directories created successfully
    """
    directories = [
        'output/depth_maps',
        'output/disparity_maps', 
        'output/point_clouds',
        'output/rectified'
    ]
    
    try:
        for directory in directories:
            full_path = os.path.join(base_path, directory)
            os.makedirs(full_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False


def save_image(image: np.ndarray, filepath: str, normalize: bool = False) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image array to save
        filepath: Output file path
        normalize: Whether to normalize the image before saving
        
    Returns:
        True if saved successfully
    """
    try:
        if normalize:
            # Normalize to 0-255 range
            if image.dtype != np.uint8:
                image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                image_norm = image_norm.astype(np.uint8)
            else:
                image_norm = image
        else:
            image_norm = image
            
        cv2.imwrite(filepath, image_norm)
        return True
    except Exception as e:
        print(f"Error saving image {filepath}: {e}")
        return False


def save_depth_data(depth: np.ndarray, filepath: str) -> bool:
    """
    Save depth data as numpy array.
    
    Args:
        depth: Depth array
        filepath: Output file path
        
    Returns:
        True if saved successfully
    """
    try:
        np.save(filepath, depth)
        return True
    except Exception as e:
        print(f"Error saving depth data {filepath}: {e}")
        return False


def load_depth_data(filepath: str) -> Optional[np.ndarray]:
    """
    Load depth data from numpy file.
    
    Args:
        filepath: Path to depth data file
        
    Returns:
        Depth array or None if loading failed
    """
    try:
        return np.load(filepath)
    except Exception as e:
        print(f"Error loading depth data {filepath}: {e}")
        return None


def calculate_fps(frame_times: List[float], window_size: int = 30) -> float:
    """
    Calculate FPS from frame timestamps.
    
    Args:
        frame_times: List of frame timestamps
        window_size: Number of frames to use for calculation
        
    Returns:
        Current FPS
    """
    if len(frame_times) < 2:
        return 0.0
    
    recent_times = frame_times[-window_size:]
    if len(recent_times) < 2:
        return 0.0
    
    time_diff = recent_times[-1] - recent_times[0]
    if time_diff == 0:
        return 0.0
    
    return (len(recent_times) - 1) / time_diff


def create_info_overlay(image: np.ndarray, fps: float, depth_range: Tuple[float, float] = None) -> np.ndarray:
    """
    Create an overlay with information on the image.
    
    Args:
        image: Input image
        fps: Current FPS
        depth_range: Optional tuple of (min_depth, max_depth) in mm
        
    Returns:
        Image with overlay
    """
    overlay = image.copy()
    
    # FPS text
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(overlay, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    # Depth range text
    if depth_range is not None:
        min_depth, max_depth = depth_range
        depth_text = f"Depth: {min_depth:.0f}-{max_depth:.0f}mm"
        cv2.putText(overlay, depth_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
    
    return overlay


def get_depth_at_point(depth_map: np.ndarray, x: int, y: int, radius: int = 5) -> Optional[float]:
    """
    Get depth value at a specific point with averaging.
    
    Args:
        depth_map: Depth map array
        x, y: Pixel coordinates
        radius: Radius for averaging
        
    Returns:
        Average depth value or None
    """
    if depth_map is None:
        return None
    
    h, w = depth_map.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    
    # Extract region around point
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    
    region = depth_map[y1:y2, x1:x2]
    valid_depths = region[region > 0]
    
    if len(valid_depths) > 0:
        return float(np.mean(valid_depths))
    else:
        return None


def create_depth_histogram(depth_map: np.ndarray, max_depth: float = 2000.0) -> np.ndarray:
    """
    Create a histogram of depth values.
    
    Args:
        depth_map: Depth map array
        max_depth: Maximum depth for histogram
        
    Returns:
        Histogram image
    """
    if depth_map is None:
        return None
    
    # Get valid depths
    valid_depths = depth_map[depth_map > 0]
    valid_depths = valid_depths[valid_depths <= max_depth]
    
    if len(valid_depths) == 0:
        return None
    
    # Create histogram
    plt.figure(figsize=(6, 4))
    plt.hist(valid_depths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Depth (mm)')
    plt.ylabel('Frequency')
    plt.title('Depth Distribution')
    plt.grid(True, alpha=0.3)
    
    # Convert plot to image
    plt.tight_layout()
    plt.savefig('temp_hist.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Read back as image
    hist_img = cv2.imread('temp_hist.png')
    os.remove('temp_hist.png')
    
    return hist_img


def resize_image_pair(left_img: np.ndarray, right_img: np.ndarray, 
                     target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize a pair of stereo images maintaining aspect ratio.
    
    Args:
        left_img: Left camera image
        right_img: Right camera image
        target_size: (width, height) target size
        
    Returns:
        Resized image pair
    """
    left_resized = cv2.resize(left_img, target_size, interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_img, target_size, interpolation=cv2.INTER_AREA)
    
    return left_resized, right_resized


def create_anaglyph(left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
    """
    Create red-cyan anaglyph from stereo pair.
    
    Args:
        left_img: Left camera image  
        right_img: Right camera image
        
    Returns:
        Anaglyph image
    """
    # Convert to grayscale if needed
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
        
    if len(right_img.shape) == 3:
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)  
    else:
        right_gray = right_img
    
    # Create anaglyph (red channel from left, cyan channels from right)
    anaglyph = np.zeros((left_gray.shape[0], left_gray.shape[1], 3), dtype=np.uint8)
    anaglyph[:, :, 2] = left_gray   # Red channel
    anaglyph[:, :, 0] = right_gray  # Blue channel
    anaglyph[:, :, 1] = right_gray  # Green channel
    
    return anaglyph


class PerformanceMonitor:
    """Monitor performance metrics for the depth estimation system."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.frame_times = []
        self.processing_times = []
        
    def add_frame_time(self, timestamp: float):
        """Add a frame timestamp."""
        self.frame_times.append(timestamp)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def add_processing_time(self, duration: float):
        """Add processing duration."""
        self.processing_times.append(duration)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return calculate_fps(self.frame_times, self.window_size)
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        return {
            'fps': self.get_fps(),
            'avg_processing_time': self.get_avg_processing_time(),
            'frame_count': len(self.frame_times)
        }


def test_utilities():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test directory creation
    if create_directories('.'):
        print("✓ Directory creation test passed")
    else:
        print("✗ Directory creation test failed")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    current_time = time.time()
    
    for i in range(10):
        monitor.add_frame_time(current_time + i * 0.033)  # ~30 FPS
        monitor.add_processing_time(0.02)  # 20ms processing
    
    stats = monitor.get_stats()
    print(f"✓ Performance monitor test: FPS={stats['fps']:.1f}, "
          f"Processing={stats['avg_processing_time']*1000:.1f}ms")


if __name__ == "__main__":
    test_utilities()