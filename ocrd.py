import os
import cv2
import numpy as np
import xml.etree.ElementTree as et

PAGE_TAG = "ns0:Page"
TEXTREGION_TAG = "ns0:TextRegion"
TEXTLINE_TAG = "ns0:TextLine"
COORDS_TAG = "ns0:Coords"
POINTS_ATTR = "points"
WIDTH_TAG = "imageWidth"
HEIGHT_TAG = "imageHeight"

NAMESPACE_GT = {
    "ns0": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

LOWER_BINARY_THRESH = 127
UPPER_BINARY_THRESH = 255

COORD_DELIM = ","

GT_LINE_COLOR = (0, 255, 0)
PRED_LINE_COLOR = (255, 0, 0)
LINES_CLOSED = True

def get_line_coords(page, namespace):
    line_list = []

    for region in page.findall(TEXTREGION_TAG, namespace):
        for line in region.findall(TEXTLINE_TAG, namespace):
            coordsElement = line.find(COORDS_TAG, namespace)
            coords = coordsElement.get(POINTS_ATTR).split()

            # Convert each coordinate value from strings to integers
            for i in range(len(coords)):
                xy_coords = coords[i].split(COORD_DELIM)

                for j in range(len(xy_coords)):
                    xy_coords[j] = int(xy_coords[j])

                coords[i] = xy_coords

            line_list.append(coords)

    return line_list

def output_image(image, gt_lines):
    # Draw each ground truth bounding box on image
    for line_coords in gt_lines:
        line_coords = np.array(line_coords, np.int32)
        image = cv2.polylines(image, [line_coords], LINES_CLOSED,
                GT_LINE_COLOR, thickness=10)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

for filename in os.listdir("OCR-D_IMG"):
    basename = filename.replace(".png", "")
    gt_file = open("OCR-D_GT/" + basename + ".xml", 'r')

    gt_page = et.parse(gt_file).getroot().find(PAGE_TAG, NAMESPACE_GT)
    gt_image_width = int(gt_page.get(WIDTH_TAG))
    gt_image_height = int(gt_page.get(HEIGHT_TAG))
    shape = (gt_image_height, gt_image_width)

    gt_lines = get_line_coords(gt_page, NAMESPACE_GT)

    image = cv2.imread("OCR-D_IMG_PNG/" + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imwrite("OCR-D_LABELS/" + filename, output_image(image, gt_lines)

    gt_file.close()
