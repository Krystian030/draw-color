# takes data saved by DRAW model and generates animations
# example usage: python plot_data.py noattn /tmp/draw/draw_data.npy
import math
import os

import matplotlib
import sys
import numpy as np

interactive=False # set to False if you want to write images to file

if not interactive:
	matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt


def xrecons_grid(X,B,A):
	"""
	plots canvas for single time step
	X is x_recons, (batch_size x img_size)
	assumes features = BxA images
	batch is assumed to be a square number
	"""
	padsize=1
	padval=.5
	ph=B+2*padsize
	pw=A+2*padsize
	batch_size=X.shape[0]
	N=int(np.sqrt(batch_size))
	X=X.reshape((N,N,B,A))
	img=np.ones((N*ph,N*pw))*padval
	for i in range(N):
		for j in range(N):
			startr=i*ph+padsize
			endr=startr+B
			startc=j*pw+padsize
			endc=startc+A
			img[startr:endr,startc:endc]=X[i,j,:,:]
	return img

if __name__ == '__main__':
	attention_points = np.load('attention_points.npy')
	prefix=sys.argv[1]
	out_file=sys.argv[2]
	C = np.load(out_file)
	T,batch_size,img_size=C.shape
	X=1.0/(1.0+np.exp(-C)) # x_recons=sigmoid(canvas)
	B=A=int(np.sqrt(img_size))
	if interactive:
		f,arr=plt.subplots(1,T)
	for t in range(T):
		img=xrecons_grid(X[t,:,:],B,A)
		if interactive:
			arr[t].matshow(img,cmap=plt.cm.gray)
			arr[t].set_xticks([])
			arr[t].set_yticks([])

		else:
			plt.matshow(img,cmap=plt.cm.gray)
			range_i = int(math.sqrt(batch_size))
			counter = 0
			for i in range(range_i):
				for j in range(range_i):
					x,y,d = attention_points[t][counter]
					x = x + (np.sqrt(img_size) + 2) * j
					y = y + (np.sqrt(img_size) + 2) * i
					# plot square with side length d
					counter += 1
					plt.plot([x-d/2,x-d/2,x+d/2,x+d/2,x-d/2],[y-d/2,y+d/2,y+d/2,y-d/2,y-d/2],'r')

			if not os.path.exists('tmp'):
				os.makedirs('tmp')
			imgname='%s_%d.png' % (prefix,t) # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
			plt.savefig('tmp/'+imgname)
			print(imgname)
