#Parameters:
#[1] - image file
#[2] - csv with row splits from solution, formatted as columns of pairs denoting row start and end

import cv2 as cv
import numpy as np
import pandas as pd
import sys

img = cv.imread(sys.argv[1], 0)
sumMatrix = np.sum(255 - img, 1)
print(sumMatrix.shape)

rRange = np.arange(25, 51)
dRange = np.arange(0, 21)


sol_rows = pd.read_csv(sys.argv[2], header=None, names=["start", "end"])

cutRowSums = []
for row_line in range(0,sol_rows.shape[0]):
    cutLine = 255 - img[sol_rows.loc[row_line].start:sol_rows.loc[row_line].end,:]
    cutRowSums.append(np.sum(cutLine, axis=1))
cutRowSums = np.array(cutRowSums)
sol_means = np.mean(cutRowSums, axis=0)
sol_var = np.var(cutRowSums, axis=0)


rdvars = np.zeros(shape=(rRange.shape[0], dRange.shape[0]))
for r in rRange:
    for d in dRange:
        rd_splits = np.array([(r*i + d) for i  in range(int(np.floor((sumMatrix.shape[0] / r))))])
       
        img_splits_rd = np.array(np.array_split(sumMatrix, rd_splits))
        img_splits_rd = np.vstack(img_splits_rd[1:-1])
        
        res = sol_means[np.linspace(0, sol_means.shape[0] - 1, r).astype(int)]
        sq = np.subtract(img_splits_rd, res)**2
        sq = np.sum(sq)
        
        rdvars[r - rRange[0], d - dRange[0]] = sq

print(rdvars.shape)

best = np.where(rdvars == rdvars.min())
R2 = best[0][0] + rRange[0]
D2 = best[1][0] + dRange[0]
print(best)
print("best r2 = " + str(R2) + ", best d2 = " + str(D2))
print("lowest var = " + str(rdvars[best[0][0], best[1][0]]))

split_indices = [(R2*(i) + D2) for i  in range(int(np.floor((sumMatrix.shape[0] / R2))))]

fname = sys.argv[1].split('.')[0] + '_splits.csv'
fout = open(fname, 'w')
for x in split_indices:
    fout.write(str(x) + '\n')
fout.close()