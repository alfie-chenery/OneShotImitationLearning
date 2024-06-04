from moviepy.editor import VideoFileClip
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = "1a-demonstration.mp4"

videoClip = VideoFileClip(dir_path + "\\" + filename)
videoClip.write_gif(dir_path + "\\" + filename[:-4] + ".gif", fps=15)