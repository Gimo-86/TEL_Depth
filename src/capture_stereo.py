"""
Stereo camera capture module for real-time depth estimation.
Handles capturing synchronized frames from two webcams.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional


class StereoCapture:
    def __init__(self, left_camera_id: int = 0, right_camera_id: int = 1, 
                 width: int = 640, height: int = 480):
        """
        Initialize stereo camera capture.
        
        Args:
            left_camera_id: ID of the left camera
            right_camera_id: ID of the right camera
            width: Frame width
            height: Frame height
        """
        self.left_camera_id = left_camera_id
        self.right_camera_id = right_camera_id
        self.width = width
        self.height = height
        
        # Initialize cameras
        self.left_cap = None
        self.right_cap = None
        self.is_initialized = False
        
    def initialize_cameras(self) -> bool:
        """Initialize both cameras."""
        try:
            # Initialize left camera
            self.left_cap = cv2.VideoCapture(self.left_camera_id)
            if not self.left_cap.isOpened():
                print(f"Failed to open left camera (ID: {self.left_camera_id})")
                return False
                
            # Initialize right camera
            self.right_cap = cv2.VideoCapture(self.right_camera_id)
            if not self.right_cap.isOpened():
                print(f"Failed to open right camera (ID: {self.right_camera_id})")
                self.left_cap.release()
                return False
            
            # Set camera properties
            self._set_camera_properties()
            
            self.is_initialized = True
            print("Stereo cameras initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing cameras: {e}")
            self.release()
            return False
    
    def _set_camera_properties(self):
        """Set common properties for both cameras."""
        properties = [
            (cv2.CAP_PROP_FRAME_WIDTH, self.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
        ]
        
        for prop, value in properties:
            self.left_cap.set(prop, value)
            self.right_cap.set(prop, value)
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture synchronized frames from both cameras.
        
        Returns:
            Tuple of (left_frame, right_frame) or (None, None) if capture fails
        """
        if not self.is_initialized:
            return None, None
        
        # Capture frames
        ret_left, frame_left = self.left_cap.read()
        ret_right, frame_right = self.right_cap.read()
        
        if ret_left and ret_right:
            return frame_left, frame_right
        else:
            return None, None
    
    def get_camera_info(self) -> dict:
        """Get information about the cameras."""
        if not self.is_initialized:
            return {}
        
        info = {
            'left_camera': {
                'width': int(self.left_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.left_cap.get(cv2.CAP_PROP_FPS)
            },
            'right_camera': {
                'width': int(self.right_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.right_cap.get(cv2.CAP_PROP_FPS)
            }
        }
        return info
    
    def release(self):
        """Release camera resources."""
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()
        self.is_initialized = False
        print("Cameras released")


def test_stereo_capture():
    """Test function for stereo capture."""
    stereo = StereoCapture()
    
    if not stereo.initialize_cameras():
        print("Failed to initialize cameras")
        return
    
    print("Camera info:", stereo.get_camera_info())
    print("Press 'q' to quit")
    
    try:
        while True:
            left_frame, right_frame = stereo.capture_frames()
            
            if left_frame is not None and right_frame is not None:
                # Display frames
                combined = np.hstack((left_frame, right_frame))
                cv2.imshow('Stereo Capture (Left | Right)', combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to capture frames")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stereo.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_stereo_capture()