#!/bin/sh

# The first script 'videoprocessing.py' accepts as an argument the path of a root folder that contains various clips. It creates a subfolder of the root folder named  that
#'Testfiles' and another two subsubfolders named 'frames' and 'images'
# These two subsubfolders contain the frames and the audio files of each clip that is processed.
#
#python3 videoprocessing.py --path_to_clip $1

# The second script receives the Testfiles folder that has been previously created as an argument and examines each pair of audio/image of it. Finally,
# it outputs the heatmaps corresponding to the visual localization of the object that emits sound.

#python3 XATZIKANELOS.py  --model_dir ./checkpoints/  --test_data_path $1/Testfiles/ --save_visualizations --alpha 0.4 --batch_size 1

# The third script soundnet.py is redirected to the audio subsubfolders that were previously created and examines each audio file.
##It outputs the predicted scene correspodning to each audio file
#
#python3 soundnet.py --path_to_clip  $1/Testfiles/audio/

cd CSAIL/places365/

python3 run_placesCNN_unified.py --path_to_image $1/Testfiles/frames/
##
#cd ..
#cd ..
#cd YOLO/yolov7/
#
#python3 detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source $1/Testfiles/frames/


#In order to run the script: ./script.sh /path_to_clips/
