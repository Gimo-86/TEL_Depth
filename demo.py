"""
Quick demo script for testing the depth estimation system.
This script provides a simple way to test the system with limited output duration.
"""

import time
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import RealtimeDepthEstimator


def run_demo(duration_seconds: int = 10):
    """
    Run a demo of the depth estimation system for a limited duration.
    
    Args:
        duration_seconds: How long to run the demo
    """
    print(f"Running depth estimation demo for {duration_seconds} seconds...")
    print("Make sure you have two cameras connected!")
    
    # Create estimator with default settings optimized for demo
    estimator = RealtimeDepthEstimator(
        left_camera_id=0,
        right_camera_id=1,
        focal_length=700.0,
        baseline=60.0,  # Assume 6cm between cameras
        stereo_method="sgbm"  # Better quality than BM
    )
    
    # Set demo parameters
    estimator.max_frames = duration_seconds * 30  # Assume 30 FPS
    estimator.record_output = False  # Don't save files in demo
    
    try:
        estimator.run()
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        print("Common issues:")
        print("- Make sure both cameras are connected")
        print("- Try different camera IDs (0,1 or 0,2 etc.)")
        print("- Check camera permissions")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick depth estimation demo")
    parser.add_argument("--duration", type=int, default=10,
                       help="Demo duration in seconds (default: 10)")
    
    args = parser.parse_args()
    
    run_demo(args.duration)
