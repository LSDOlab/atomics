import subprocess
from subprocess import call
# import paraview_script
# from paraview.simple import *
# from para import para_script
# import importlib


	
def make_mov(png_filename, movie_filename):
        # for i in range num_nodes:
        


    cmd = 'ffmpeg -i {} -q:v 1 -vcodec mpeg4 {}.avi'.format(png_filename, movie_filename)
    # cmd = 'ffmpeg -i {} -vcodec mpeg4 {}.avi'.format(png_filename, movie_filename)
    call(cmd.split())
    cmd = 'ffmpeg -i {}.avi -acodec libmp3lame -ab 384 {}.mov'.format(movie_filename, movie_filename)
    call(cmd.split())
	
def make_mp4(png_filename, movie_filename):


    # real time 130fps; two times slower 65fps; four times lower 37.5fps
    bashCommand = "ffmpeg -f image2 -r 65 -i {} -vcodec libx264 -y {}.mp4".format(png_filename, movie_filename)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

png_filename = 'density%1d.png'
movie_filename = 'mov'
make_mov(png_filename, movie_filename)
make_mp4(png_filename, movie_filename)