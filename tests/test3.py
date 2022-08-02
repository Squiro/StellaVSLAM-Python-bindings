'''
This test runs stella_vslam on video.
It only shows the pose matrix and the video feed.
It doesn't show map nor features.

Command line example:
python3 test2.py -c ./aist_living_lab_1/config.yaml -m ./aist_living_lab_1/video.mp4

Valid pose only after initialization.
You can check the pose is valid when last row is 0,0,0,1 .

stella_vslam is not bug free.  Sometimes it crashes (segmentation fault) right after map initialization.
It's not a bindings bug.  Try again - many times - until succeeding.

Map save not tested.
'''

import stella_vslam
import cv2 as cv
import sys
import argparse
from threading import Thread

def run_slam():
    global frameShowFactor
    global video
    global SLAM

    pose = []
    timestamp = 0.0
    print("Entering the video feed loop.")
    print("You should soon see the video in a window, and the 4x4 pose matrix on this terminal.")
    print("ESC to quit (focus on window: click on feeding frame window, then press ESC).")
    is_not_end = True   
    
    while(is_not_end):
        is_not_end, frame = video.read()    
        if(frame.size):
            # cv.imshow("Feeding frame", cv.resize(frame, None, fx=frameShowFactor, fy=frameShowFactor))        
            retVal, pose = SLAM.feed_monocular_frame(frame, timestamp) # fake timestamp to keep it simple
        if((timestamp % 30) == 0):
            print("Timestamp", timestamp, ", Pose:")
            
            # Format pose matrix with only a few decimals
            for row in pose:
                for data in row:
                    sys.stdout.write('{:9.1f}'.format(data))
                print()
        timestamp += 1
        key = cv.waitKey(1)  # Needed to refresh imshow window
        if (key == 27):
            # ESC, finish
            break

# Some arguments from run_video_slam.cc
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vocab", help="vocabulary file path", default="./orb_vocab.dbow2")
parser.add_argument("-m", "--video", help="video file path", required=True)
parser.add_argument("-c", "--config", help="config file path", default="./config.yaml")
parser.add_argument("-p", "--map_db", help="store a map database at this path after SLAM")
parser.add_argument("-f", "--factor", help="scale factor to show video in window - doesn't affect stella_vslam", default=0.5, type=float)
args = parser.parse_args()

config = stella_vslam.config(config_file_path=args.config)

SLAM = stella_vslam.system(cfg=config, vocab_file_path=args.vocab)
VIEWER = stella_vslam.viewer(config.yaml_node_['PangolinViewer'], SLAM)

SLAM.startup()
print("stella_vslam up and operational.")

video = cv.VideoCapture(args.video)

frameShowFactor = args.factor

slamThreadInstance = Thread(target=run_slam)

slamThreadInstance.start()
VIEWER.run()
slamThreadInstance.join()

SLAM.save_map_database(args.map_db)
SLAM.shutdown()