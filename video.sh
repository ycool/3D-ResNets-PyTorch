#! /bin/bash

set -x 

f=datasets/hmdb51/run/Two_Towers_1_run_f_cm_np1_fr_med_3.avi
ffprobe -v error -select_streams v:0  \
	-of default=noprint_wrappers=1:nokey=1 -show_entries  \
	stream=width,height,avg_frame_rate,duration \
	$f


ffmpeg -i $f \
       -vf scale=-1:240 -threads 1 '/tmp/image_%05d.jpg'


