import moviepy.editor as mp
import os
import cv2
import argparse
from glob import glob
from pathlib import Path

def extract_frames_and_audio(path):

    print(path)
    path_ss = path.replace(".mp4","")
    path_s = path_ss.replace(".", "")
    print(path_s)
    y = Path(path).stem
    # Read the video from specified path
    cam = cv2.VideoCapture(path)
    fps = cam.get(cv2.CAP_PROP_FPS)

    if not os.path.exists(path_s):
        os.makedirs(path_s + './frames/')
        os.makedirs(path_s + './audio/')



    currentframe = 0
    while (True):
    # reading from frame
        ret, frame = cam.read()

        if ret:
            name = path_s + './frames/' + y + str(currentframe) + '.jpg'
        # writing the extracted images
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break


    clip = mp.VideoFileClip(path)


    clip.audio.write_audiofile(path_s + './audio/'+ y + r".wav")

    cam.release()
    cv2.destroyAllWindows()
    path_sss = path_ss + './'
    os.rename(path_sss,y)

    return

#### Parsing the arguments
parser = argparse.ArgumentParser()

parser.add_argument('--path_to_clip',metavar='path_to_clip',  default='', type=str, help='Path of the clip')
args= parser.parse_args()

path = args.path_to_clip



clips = glob(path + '*mp4') + glob(path + '*avi')
for c in clips:
    extract_frames_and_audio(c)

