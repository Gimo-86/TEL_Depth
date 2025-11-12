"""
Real-time depth estimation from two webcams.
Main application that integrates stereo capture, disparity computation, and depth visualization.
"""

import cv2
import numpy as np
import time
import os
import argparse
from typing import Optional

from src.capture_stereo import StereoCapture
from src.compute_disparity import DisparityComputer
from src.depth_estimation import DepthEstimator
from src.visualize_depth import DepthVisualizer
from src.utils import create_directories, save_image, save_depth_data, PerformanceMonitor


class RealtimeDepthEstimator:
    def __init__(self, left_camera_id: int = 0, right_camera_id: int = 1,
                 focal_length: float = 700.0, baseline: float = 60.0,
                 stereo_method: str = "sgbm"):
        """
        Initialize real-time depth estimator.
        
        Args:
            left_camera_id: ID of left camera
            right_camera_id: ID of right camera
            focal_length: Camera focal length in pixels
            baseline: Distance between cameras in mm
            stereo_method: Stereo matching method ('bm' or 'sgbm')
        """
        self.left_camera_id = left_camera_id
        self.right_camera_id = right_camera_id
        
        # Initialize components
        self.stereo_capture = StereoCapture(left_camera_id, right_camera_id)
        self.disparity_computer = DisparityComputer(method=stereo_method)
        self.depth_estimator = DepthEstimator(focal_length, baseline)
        self.visualizer = DepthVisualizer()
        self.performance_monitor = PerformanceMonitor()
        
        # Recording settings
        self.record_output = False
        self.output_dir = "output"
        self.frame_count = 0
        self.max_frames = None  # Set to limit recording duration
        
        # Processing settings
        self.enable_rectification = True
        self.enable_post_processing = True
        
        print(f"Initialized depth estimator:")
        print(f"  Cameras: {left_camera_id}, {right_camera_id}")
        print(f"  Focal length: {focal_length}px")
        print(f"  Baseline: {baseline}mm")
        print(f"  Stereo method: {stereo_method}")
    
    def initialize(self) -> bool:
        """Initialize all components."""
        print("Initializing components...")
        
        # Create output directories
        if not create_directories(self.output_dir):
            print("Warning: Failed to create output directories")
        
        # Initialize cameras
        if not self.stereo_capture.initialize_cameras():
            print("Failed to initialize cameras")
            return False
        
        # Get camera info and setup calibration
        camera_info = self.stereo_capture.get_camera_info()
        print("Camera info:", camera_info)
        
        left_cam = camera_info['left_camera']
        image_size = (left_cam['width'], left_cam['height'])
        
        # Try to load calibration, otherwise use defaults
        calibration_dir = os.path.join("calibration", "camera_param")
        if not self.depth_estimator.load_calibration(calibration_dir):
            print("Using default calibration parameters")
            self.depth_estimator.setup_default_calibration(image_size)
        
        # Setup rectification if enabled
        if self.enable_rectification:
            if not self.depth_estimator.compute_rectification_maps(image_size):
                print("Warning: Failed to compute rectification maps")
                self.enable_rectification = False
        
        print("Initialization complete!")
        return True
    
    def process_frame_pair(self, left_frame: np.ndarray, 
                          right_frame: np.ndarray) -> tuple:
        """
        Process a pair of stereo frames to compute depth.
        
        Args:
            left_frame: Left camera frame
            right_frame: Right camera frame
            
        Returns:
            Tuple of (depth_map, disparity_map, depth_visualization, disparity_visualization)
        """
        start_time = time.time()
        
        # Rectify images if enabled
        if self.enable_rectification:
            left_rect, right_rect = self.depth_estimator.rectify_images(left_frame, right_frame)
        else:
            left_rect, right_rect = left_frame, right_frame
        
        # Compute disparity
        disparity = self.disparity_computer.compute_disparity(left_rect, right_rect)
        
        if disparity is None:
            return None, None, None, None
        
        # Post-process disparity if enabled
        if self.enable_post_processing:
            disparity = self.disparity_computer.post_process_disparity(disparity)
        
        # Convert to depth
        depth_map = self.depth_estimator.disparity_to_depth(disparity)
        
        # Create visualizations
        depth_vis = self.depth_estimator.visualize_depth(depth_map)
        disparity_vis = self.disparity_computer.visualize_disparity(disparity)
        
        # Record processing time
        processing_time = time.time() - start_time
        self.performance_monitor.add_processing_time(processing_time)
        
        return depth_map, disparity, depth_vis, disparity_vis
    
    def save_frame_data(self, left_frame: np.ndarray, right_frame: np.ndarray,
                       depth_map: np.ndarray, disparity: np.ndarray,
                       depth_vis: np.ndarray, disparity_vis: np.ndarray):
        """Save frame data to disk."""
        if not self.record_output:
            return
        
        timestamp = f"{self.frame_count:06d}"
        
        # Save original images
        save_image(left_frame, f"{self.output_dir}/frames/left_{timestamp}.jpg")
        save_image(right_frame, f"{self.output_dir}/frames/right_{timestamp}.jpg")
        
        # Save depth data
        if depth_map is not None:
            save_depth_data(depth_map, f"{self.output_dir}/depth_maps/depth_{timestamp}.npy")
            save_image(depth_vis, f"{self.output_dir}/depth_maps/depth_vis_{timestamp}.jpg")
        
        # Save disparity data
        if disparity is not None:
            save_depth_data(disparity, f"{self.output_dir}/disparity_maps/disparity_{timestamp}.npy")
            save_image(disparity_vis, f"{self.output_dir}/disparity_maps/disparity_vis_{timestamp}.jpg")
    
    def run(self):
        """Run the real-time depth estimation."""
        if not self.initialize():
            return
        
        print("\nStarting real-time depth estimation...")
        print("Controls:")
        print("  ESC: Quit")
        print("  R: Toggle recording")  
        print("  P: Toggle post-processing")
        print("  T: Toggle rectification")
        print("  See visualization window for more controls")
        
        try:
            while True:
                # Capture frames
                left_frame, right_frame = self.stereo_capture.capture_frames()
                
                if left_frame is None or right_frame is None:
                    print("Failed to capture frames")
                    break
                
                # Process frames
                depth_map, disparity, depth_vis, disparity_vis = self.process_frame_pair(
                    left_frame, right_frame
                )
                
                # Update display
                if not self.visualizer.update_display(
                    left_frame, right_frame, depth_map, disparity, depth_vis, disparity_vis
                ):
                    break
                
                # Save data if recording
                if self.record_output:
                    self.save_frame_data(
                        left_frame, right_frame, depth_map, disparity, depth_vis, disparity_vis
                    )
                
                self.frame_count += 1
                
                # Check frame limit
                if self.max_frames and self.frame_count >= self.max_frames:
                    print(f"Reached maximum frame limit: {self.max_frames}")
                    break
                
                # Handle global key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r') or key == ord('R'):
                    self.record_output = not self.record_output
                    status = "ON" if self.record_output else "OFF"
                    print(f"Recording: {status}")
                elif key == ord('p') or key == ord('P'):
                    self.enable_post_processing = not self.enable_post_processing
                    status = "ON" if self.enable_post_processing else "OFF"
                    print(f"Post-processing: {status}")
                elif key == ord('t') or key == ord('T'):
                    self.enable_rectification = not self.enable_rectification
                    status = "ON" if self.enable_rectification else "OFF"
                    print(f"Rectification: {status}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        # Print final statistics
        stats = self.performance_monitor.get_stats()
        print(f"Final statistics:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Average FPS: {stats['fps']:.1f}")
        print(f"  Average processing time: {stats['avg_processing_time']*1000:.1f}ms")
        
        # Release resources
        self.stereo_capture.release()
        self.visualizer.cleanup()
        
        print("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time stereo depth estimation")
    parser.add_argument("--left-camera", type=int, default=0, 
                       help="Left camera ID (default: 0)")
    parser.add_argument("--right-camera", type=int, default=1,
                       help="Right camera ID (default: 1)")
    parser.add_argument("--focal-length", type=float, default=700.0,
                       help="Camera focal length in pixels (default: 700)")
    parser.add_argument("--baseline", type=float, default=60.0,
                       help="Distance between cameras in mm (default: 60)")
    parser.add_argument("--stereo-method", choices=["bm", "sgbm"], default="sgbm",
                       help="Stereo matching method (default: sgbm)")
    parser.add_argument("--max-frames", type=int, 
                       help="Maximum number of frames to process")
    parser.add_argument("--record", action="store_true",
                       help="Start recording immediately")
    
    args = parser.parse_args()
    
    # Create and run depth estimator
    estimator = RealtimeDepthEstimator(
        left_camera_id=args.left_camera,
        right_camera_id=args.right_camera,
        focal_length=args.focal_length,
        baseline=args.baseline,
        stereo_method=args.stereo_method
    )
    
    estimator.max_frames = args.max_frames
    estimator.record_output = args.record
    
    estimator.run()


if __name__ == "__main__":
    main()