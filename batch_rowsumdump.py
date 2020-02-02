import cv2
import numpy as np
import os
import sys


directory = sys.argv[1]
outputDir = os.path.join(os.path.abspath(directory), "rowsumdump")

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

for filename in os.listdir(directory):
    if filename.endswith(".tif"):
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        print(im.shape)

        im = 1-im/255
        sums = np.sum(im, axis=1)
        print(len(sums))
        print(sums.shape)
        print(np.max(im, axis=1))


        #fname = sys.argv[1].split('.')[0]+'.csv'
        fname = os.path.basename(filename).replace('.tif', '.csv')
        fout = open(os.path.join(outputDir, fname),'w')
        for x in sums:
            fout.write(str(x)+'\n')
        fout.close()
