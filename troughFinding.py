#Parameters:
#[1] - image file

import cv2 as cv
import numpy as np
import pandas as pd
import scipy.signal as sig

img = cv.imread(sys.argv[1], 0)
sumMatrix = np.sum(255 - img, 1)

smooth = sig.savgol_filter(sumMatrix, 51, 3)
troughs, _ = sig.find_peaks(-smooth)

heights = np.zeros(len(troughs) - 1)
for i in np.arange(0,len(troughs) - 1 ):
    start = troughs[i]
    end = troughs[i+1]
    
    cutLine = img[start:end,:]
    heights[i] = end-start

trimHeights = abs(heights - np.median(heights)) < 1.5 * np.std(heights)
print(trimHeights)
print(trimHeights.shape)

trimTroughs = troughs
i = 0
while i < trimTroughs.shape[0] - 1:
    if(not trimHeights[i]):
        trimTroughs = np.delete(trimTroughs, i+1, 0)
        trimHeights = np.delete(trimHeights, i, 0)
        continue
    start = trimTroughs[i]
    end = trimTroughs[i+1]
    
    i+=1
        
print(trimTroughs.shape)