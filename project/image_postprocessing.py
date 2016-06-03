from PIL import Image, ImageDraw
import sys

f = open(sys.argv[2],'r')
width = int(sys.argv[3])
height = int(sys.argv[4])
im = Image.new("RGB",(width, height))


row = 1
line = f.readline()
rowdata = line.split(",")
if '\n' in rowdata:
	rowdata.remove('\n')
rgb_list = [(int(pixel), int(pixel), int(pixel)) for pixel in rowdata]
im.putdata(rgb_list)
im.save(sys.argv[1])




