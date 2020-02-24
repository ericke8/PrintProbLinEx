'''
Line extraction tool using Tesseract 4.0.

Author: Jason Vega
Email: jvega@ucsd.edu
'''

from tesserocr import PyTessBaseAPI
from PIL import Image
import os
import sys
import time
import getopt
import xml.etree.ElementTree as et

IMAGE_EXTENSION = ".tif"
FILE_EXTENSION = ".xml"

NS_PAGE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 " + \
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/" + \
        "pagecontent.xsd"

RESULTS_FILE = "time.csv"

ROOT_TAG = "{" + NS_PAGE + "}PcGts"
PAGE_TAG = "{" + NS_PAGE + "}Page"
TEXT_REGION_TAG = "{" + NS_PAGE + "}TextRegion"
TEXT_LINE_TAG = "{" + NS_PAGE + "}TextLine"
COORDS_TAG = "{" + NS_PAGE + "}Coords"
SCHEMA_LOCATION_ATTR ="{" + NS_XSI + "}schemaLocation"

OPTIONS = "t"
TIME_OPT = "-t"
ARGS = 2

'''
Parse command line arguments and Tesseract 4.0 line extraction.

argv: command line arguments.
'''
def main(argv):
    image_dir = ""
    out_dir = ""
    get_time = False

    # Parse command line options
    try:
        opts, args = getopt.getopt(argv, OPTIONS)

        if len(args) != ARGS:
            raise getopt.GetoptError("Invalid number of arguments specified.")

        for opt, arg in opts:
            if opt == TIME_OPT:
                get_time = True
            else:
                raise getopt.GetoptError("Unrecognized option passed.")

        image_dir = args[0] + "/" if args[0][-1] != "/" else args[0]
        out_dir = args[1] + "/" if args[1][-1] != "/" else args[1]
    except getopt.GetoptError:
        print("Usage: python3 tesseract.py [OPTION] image_dir out_dir")
        print("image_dir - the directory containing the document images to " + \
                "segment.")
        print("out_dir - the directory to output segmentation results to.\n")
        print("Optional flags:\n")
        print("-t - record segmentation time for each document. Results " + \
                "are written to a file in out_dir.")

        sys.exit()

    results_file = None

    if get_time:
        try:
            results_file = open(out_dir + RESULTS_FILE, 'w+')
            results_file.write("File,Lines,Time\n")
        except FileNotFoundError:
            print("Could not open " + out_dir + RESULTS_FILE + " for " + \
                    "writing. Check that the specified output directory " + \
                    "exists.\n")
            get_time = False

    # Extract lines using Tesseract
    with PyTessBaseAPI() as api:
        file_names = os.listdir(image_dir)
        start = 0
        end = 0

        if file_names:
            file_names.sort()

        for image_file in file_names:
            image = Image.open(image_dir + image_file)
            output_file = out_dir + \
                        image_file.replace(IMAGE_EXTENSION, FILE_EXTENSION)
            width, height = image.size

            root = et.Element(ROOT_TAG)
            root.set(SCHEMA_LOCATION_ATTR, SCHEMA)
            page = et.SubElement(root, PAGE_TAG, imageWidth=str(width),
                    imageHeight=str(height))
            region = et.SubElement(page, TEXT_REGION_TAG)

            api.SetImageFile(image_dir + image_file)

            print("Extracting lines from " + image_file + "...")

            if get_time:
                start = time.time_ns()

            lines = api.GetTextlines()

            if get_time:
                end = time.time_ns()
                results_file.write(image_file + "," + str(len(lines)) + "," + \
                        str(end - start) + "\n")

            for line in lines:
                x_coord = line[1]['x']
                y_coord = line[1]['y']
                line_width = line[1]['w']
                line_height = line[1]['h']

                top_left = str(x_coord) + "," + str(y_coord)
                top_right = str(x_coord + line_width) + "," + str(y_coord)
                bottom_right = str(x_coord + line_width) + "," + \
                        str(y_coord + line_height)
                bottom_left = str(x_coord) + "," + str(y_coord + line_height)

                pointsAttr = top_left + " " + top_right + " " + bottom_right + \
                    " " + bottom_left

                line = et.SubElement(region, TEXT_LINE_TAG)
                et.SubElement(line, COORDS_TAG, points=pointsAttr)

            tree = et.ElementTree(root)

            try:
                tree.write(output_file)
            except FileNotFoundError:
                print("Could not open " + output_file + " for writing. " + \
                        "Line segmentation results not saved.")

    if get_time:
        results_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
