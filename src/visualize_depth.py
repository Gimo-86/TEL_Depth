"""
Depth visualization module for real-time display and analysis.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, List, Dict
from src.utils import create_info_overlay, get_depth_at_point, PerformanceMonitor


class DepthVisualizer:
    def __init__(self, window_size: Tuple[int, int] = (1280, 720)):
        """
        Initialize depth visualizer.
        
        Args:
            window_size: (width, height) of display window
        """
        self.window_size = window_size
        self.display_mode = "quad"  # "quad", "depth", "stereo", "anaglyph"
        self.show_info = True
        self.click_pos = None
        self.performance_monitor = PerformanceMonitor()
        
        # Color settings
        self.colormap = cv2.COLORMAP_JET
        self.max_depth_display = 2000.0  # mm
        
        # Mouse callback for depth measurement
        cv2.namedWindow('Depth Estimation', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Depth Estimation', self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for depth measurement."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Adjust coordinates based on display mode and layout
            if self.display_mode == "quad":
                # Determine which quadrant was clicked
                h, w = self.window_size[1] // 2, self.window_size[0] // 2
                if x < w and y < h:
                    # Top-left (original left)
                    self.click_pos = (x, y, "left")
                elif x >= w and y < h:
                    # Top-right (original right)
                    self.click_pos = (x - w, y, "right")
                elif x < w and y >= h:
                    # Bottom-left (depth)
                    self.click_pos = (x, y - h, "depth")
                elif x >= w and y >= h:
                    # Bottom-right (disparity)
                    self.click_pos = (x - w, y - h, "disparity")
            else:
                self.click_pos = (x, y, self.display_mode)
    
    def create_quad_view(self, left_img: np.ndarray, right_img: np.ndarray,
                        depth_img: np.ndarray, disparity_img: np.ndarray) -> np.ndarray:
        """
        Create a quad view layout.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image  
            depth_img: Depth visualization
            disparity_img: Disparity visualization
            
        Returns:
            Combined quad view image
        """
        h, w = self.window_size[1] // 2, self.window_size[0] // 2
        
        # Resize all images to quarter size
        left_resized = cv2.resize(left_img, (w, h))
        right_resized = cv2.resize(right_img, (w, h))
        
        if depth_img is not None:
            depth_resized = cv2.resize(depth_img, (w, h))
        else:
            depth_resized = np.zeros((h, w, 3), dtype=np.uint8)
            
        if disparity_img is not None:
            disparity_resized = cv2.resize(disparity_img, (w, h))
        else:
            disparity_resized = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add labels
        cv2.putText(left_resized, "Left Camera", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(right_resized, "Right Camera", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_resized, "Depth Map", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disparity_resized, "Disparity Map", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine into quad layout
        top_row = np.hstack([left_resized, right_resized])
        bottom_row = np.hstack([depth_resized, disparity_resized])
        quad_view = np.vstack([top_row, bottom_row])
        
        return quad_view
    
    def update_display(self, left_img: np.ndarray, right_img: np.ndarray,
                      depth_map: np.ndarray = None, disparity_map: np.ndarray = None,
                      depth_img: np.ndarray = None, disparity_img: np.ndarray = None) -> bool:
        """
        Update the display with new images.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            depth_map: Raw depth map for measurements
            disparity_map: Raw disparity map
            depth_img: Depth visualization image
            disparity_img: Disparity visualization image
            
        Returns:
            True if display should continue, False to quit
        """
        # Update performance monitoring
        self.performance_monitor.add_frame_time(time.time())
        
        # Create main display based on mode
        if self.display_mode == "quad":
            display_img = self.create_quad_view(left_img, right_img, depth_img, disparity_img)
        elif self.display_mode == "depth":
            display_img = depth_img if depth_img is not None else left_img
        elif self.display_mode == "stereo":
            display_img = np.hstack([left_img, right_img])
        elif self.display_mode == "anaglyph":
            display_img = self._create_anaglyph(left_img, right_img)
        else:
            display_img = left_img
        
        # Add information overlay
        if self.show_info:
            display_img = self._add_info_overlay(display_img, depth_map)
        
        # Show depth measurement at click position
        if self.click_pos is not None:
            display_img = self._add_depth_measurement(display_img, depth_map)
        
        # Display the image
        cv2.imshow('Depth Estimation', display_img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        return self._handle_keypress(key)
    
    def _create_anaglyph(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """Create red-cyan anaglyph."""
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            
        if len(right_img.shape) == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img
        
        # Create anaglyph
        anaglyph = np.zeros((left_gray.shape[0], left_gray.shape[1], 3), dtype=np.uint8)
        anaglyph[:, :, 2] = left_gray   # Red channel
        anaglyph[:, :, 0] = right_gray  # Blue channel  
        anaglyph[:, :, 1] = right_gray  # Green channel
        
        return anaglyph
    
    def _add_info_overlay(self, image: np.ndarray, depth_map: np.ndarray = None) -> np.ndarray:
        """Add information overlay to image."""
        overlay = image.copy()
        
        # Performance info
        fps = self.performance_monitor.get_fps()
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display mode
        cv2.putText(overlay, f"Mode: {self.display_mode.upper()}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Depth statistics
        if depth_map is not None:
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) > 0:
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)
                mean_depth = np.mean(valid_depths)
                
                cv2.putText(overlay, f"Depth: {min_depth:.0f}-{max_depth:.0f}mm", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"Mean: {mean_depth:.0f}mm", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Controls help
        help_text = [
            "Controls:",
            "Q/1: Quad view",
            "D/2: Depth only", 
            "S/3: Stereo view",
            "A/4: Anaglyph",
            "I: Toggle info",
            "ESC: Quit"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(overlay, text, (image.shape[1] - 200, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def _add_depth_measurement(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Add depth measurement at click position."""
        if self.click_pos is None or depth_map is None:
            return image
        
        x, y, region = self.click_pos
        
        # Adjust coordinates for quad view
        if self.display_mode == "quad" and region == "depth":
            # Scale coordinates back to original depth map size
            h_scale = depth_map.shape[0] / (self.window_size[1] // 2)
            w_scale = depth_map.shape[1] / (self.window_size[0] // 2)
            orig_x = int(x * w_scale)
            orig_y = int(y * h_scale)
        else:
            orig_x, orig_y = x, y
        
        # Get depth value
        depth_value = get_depth_at_point(depth_map, orig_x, orig_y)
        
        if depth_value is not None:
            # Draw crosshair
            color = (0, 255, 255)  # Yellow
            cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 20, 2)
            
            # Add text
            text = f"Depth: {depth_value:.0f}mm"
            cv2.putText(image, text, (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image
    
    def _handle_keypress(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code
            
        Returns:
            True to continue, False to quit
        """
        if key == 27:  # ESC
            return False
        elif key in [ord('q'), ord('Q'), ord('1')]:
            self.display_mode = "quad"
        elif key in [ord('d'), ord('D'), ord('2')]:
            self.display_mode = "depth"  
        elif key in [ord('s'), ord('S'), ord('3')]:
            self.display_mode = "stereo"
        elif key in [ord('a'), ord('A'), ord('4')]:
            self.display_mode = "anaglyph"
        elif key in [ord('i'), ord('I')]:
            self.show_info = not self.show_info
        elif key in [ord('c'), ord('C')]:
            self.click_pos = None  # Clear click position
        elif key == ord('+'):
            self.max_depth_display = min(5000, self.max_depth_display + 200)
        elif key == ord('-'):
            self.max_depth_display = max(500, self.max_depth_display - 200)
        
        return True
    
    def cleanup(self):
        """Clean up visualization resources."""
        cv2.destroyAllWindows()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.performance_monitor.get_stats()


def test_visualizer():
    """Test the depth visualizer."""
    print("Testing depth visualizer...")
    
    # Create test images
    height, width = 480, 640
    left_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    depth_map = np.random.randint(500, 2000, (height, width), dtype=np.float32)
    
    # Create visualizations
    depth_vis = cv2.applyColorMap(
        cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # Test visualizer
    visualizer = DepthVisualizer()
    
    print("Showing test images. Press ESC to quit, other keys to test modes.")
    
    while True:
        if not visualizer.update_display(left_img, right_img, depth_map, None, depth_vis, depth_vis):
            break
    
    visualizer.cleanup()
    print("Visualizer test completed")


if __name__ == "__main__":
    test_visualizer()