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
    - `python rowsumdump.py FILENAME`
    
It will output a file with `FILENAME.csv`, which contains the rowsums

2. Run `line_splines.R` with the rowsums csv file
    - `Rscript ./line_splines.R ./FILENAME.csv`
This will output another file named `FILENAME_lines.csv`, which contains the start and end indexes for each line

3. Open up `cutLines.ipynb` and run through the cells, changing file names when necessary
