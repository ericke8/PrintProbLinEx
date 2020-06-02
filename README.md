# PrintProbLinEx
Authors: Eric Ke, Jason Vega, Annie Wong, Nidhi Giridhar

Code repo for proposed solution #1 toward text line extraction

Clone the repo:  
`git clone https://github.com/ericke8/PrintProbLinEx.git`

## Setup:
- Make sure to have R and Python 3 installed
- Order so far:
    - rowsumdump.py
    - line_splines.R
    - cutLines.ipynb
    
### Python packages:
- `cv2` (OpenCV)
- `matplotlib`
- `pandas`
- `numpy`

### R packages:
- `mgcv`
- `dplyr`


## Instructions to run:
1. Move into the repo directory and run `rowsumdump.py` with the `.tif` image file of your choice
    - `cd PrintProbLinEx`
    - `python rowsumdump.py NameOfTheFileYouChose.tif`
    
It will output a file with `NameOfTheFileYouChose.csv`, which contains the rowsums

2. Run `line_splines.R` with the rowsums csv file
    - `Rscript ./line_splines.R ./NameOfTheFileYouChose.csv`  
    
This will output another file named `NameOfTheFileYouChose_lines.csv`, which contains the start and end indexes for each line

3. Open up `cutLines.ipynb` and run through the cells, changing file names when necessary

# Running in batches:
Same as above, but using `batch_rowsumdump.py` with a directory as input (where your images are)

This will create a new folder called `rowsumdump` in the image directory with all the csv

Then, use `./batch_line_splines.R` with the `rowsumdump` directory as input, so it processes all the images into cuts

## Method 2: Trough Finding:
1. 

## Index

### Line Splines (coded in R and python, old approach not made by us):
**rowsumdump.py** - Script to dump the row sums of an image for use in line splines \
**line_splines.R** - Script to fit a curve onto the histogram projection and output line cuts in a CSV format of start, end \
**batchrowsumdump.py** - Tool to run rowsumdump on a batch of images \
**batchLineSplines.R** - Tool to run line splines on a batch of images and output line CSV files \
**ImageLines.ipynb** - Using a lines CSV file (formatted as pairs of start,end) to display the cut lines of an image \
**evaluateLineSplines.ipynb**

### Fixed line height, R and D statistical approach:
**RDSolMeans.ipynb** - Using an image, finds the histogram projections and visuals, as well as experiments with fixed line heights and offsets, displays the cuts on the image \
**cutLines.ipynb** - Using an image, tries to find the best line height and offset to minimize distances from ground truth, displays visuals \
**meanHistogram.ipyn**b - Using an image and a lines CSV (formatted as list of line splits), displays the mean histogram of all lines \
**rowImages.ipynb** - Using an image and a lines CSV file (formatted as a list of line split locations), display the individual row images

### Trough Finding method:
**TroughFinding.ipynb** - Using an image, applies the trough finding method and visualizes the cuts (better than RDSolMeans) \
**troughFinding.py** - Command line tool to apply the trough finding method and output (incomplete) \
**evaluateTroughFinding.ipynb** - Using an image, applies the trough finding method and compares it with the ground truth cuts with visualization
**evaluateTroughFindingBoxes.ipynb**

### Tesseract:
**tesseract.py** - Line extraction using Tesseract 4

### dhSegment:
**dhSegment folder** - UNUSED
**label.py** - Command line tool for producing color masks of images taking PAGEXML as input
**getSplits.py** - Command line tool for randomly splitting data into portions (for train, val, and test)

