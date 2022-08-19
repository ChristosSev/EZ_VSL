import moviepy.editor as mp
import os
import cv2
import argparse
from glob import glob
import pathlib
from pathlib import Path
import math
import time


tic = time.perf_counter()


stem_list = []

#### Parsing the arguments
parser = argparse.ArgumentParser()

parser.add_argument('--path_to_clip',metavar='path_to_clip',  default='', type=str, help='Path of the clip')
args= parser.parse_args()

path = args.path_to_clip
path_2 = args.path_to_clip

def extract_frames_and_audio(path):

    #print(path)
    path_ss = path.replace(".mp4","")
    #print(path_ss)
    path_s = path_ss.replace(".", "")


    path_final = os.path.dirname((path_s))
    path_finale = path_final+ '/'
   # print(path_finale)

    y = Path(path).stem
    # Read the video from specified path
    cam = cv2.VideoCapture(path)
    # fps = cam.get(cv2.CAP_PROP_FPS)


    os.makedirs(path_finale + 'Testfiles',exist_ok=True)
   # print("aaaaaaaa")
    os.makedirs(path_finale + 'Testfiles/frames', exist_ok=True)
    os.makedirs(path_finale + 'Testfiles/audio', exist_ok=True)
        # os.makedirs(path_s + 'audio')

    lastframe = 0
    frame_list = []
    frame_names = []
    currentframe = 0

    while (True):

        ret, frame = cam.read()
        frame_list.append(frame)

        if ret:
            name = path_finale + 'Testfiles/frames/' + y + str(currentframe) + '.jpg'
            frame_names.append(name)
            currentframe += 1
        else:
            break


    center_frame = frame_list[math.ceil(len(frame_list)/2)]

    center_frame_name = path_finale + 'Testfiles/frames/' + y + '.jpg'


    cv2.imwrite(center_frame_name, center_frame)

    clip = mp.VideoFileClip(path)
    clip.audio.write_audiofile(path_finale + 'Testfiles/audio/'+ y + r".wav")

    cam.release()
    cv2.destroyAllWindows()
    # path_sss = path_ss + './'
    # os.rename(path_sss,y)
    stem_list.append(y)
   # print(stem_list)

    return



clips = glob(path + '*mp4') + glob(path + '*avi') + glob(path + '*MP4')
for c in clips:
    extract_frames_and_audio(c)

toc = time.perf_counter()
print(f"total time to run the script in {toc - tic:0.4f} seconds")
# for i in stem_list:




