from PIL import Image 
import sys

im = Image.open(sys.argv[1],'r')
pix = list(im.getdata())
print im.size
if (im.mode == "L"):
	pix_bw = pix
elif (im.mode == "RGB"):
	pix_bw = [ item[0] for item in pix ]
else: 
	print "unhandle im mode :", im.mode 
f = open('output_image.txt','w')
for pixel in pix_bw:
	f.write(str(pixel))
	f.write(',')
f.write('\n')
f.close()