import os
import argparse
from utils import *

parser = argparse.ArgumentParser(description="Facial emotion detector.")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--img', type=str, help='Path to the input image file')
group.add_argument('--vid', type=str, help='Path to the input video file')
group.add_argument('--cam', action='store_true', help='Use real-time data')

args = parser.parse_args()

if args.img:
    print(f"Processing image from: {args.img}")
    if not os.path.exists(args.img):
        print("File does not exists in specified location")
    else:
        detect_picture(args.img)

elif args.vid:
    print(f"Processing video from: {args.vid}")
    print(args.vid)
    if not os.path.exists(args.vid):
        print("File does not exists in specified location")
    else:
        detect_video(args.vid)

elif args.cam:
    print("Processing input from camera")
    detect_video(0)
