'''
Tool for producing colored masks of ground truth data stored in the PAGE XML 
format. Each mask contains the top and bottom lines of each text line bounding 
box in the document image. Overlapping lines are colored as the combination of 
the top and bottom line colors.

Author: Jason Vega
Email: jasonvega14@yahoo.com
'''

import os
import sys
import argparse
import cv2
import numpy as np
import xml.etree.ElementTree as et
from math import atan2

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

COORD_DELIM = ","

THICKNESS = 10
TOP_LINE_COLOR = (0, 255, 0)
BOTTOM_LINE_COLOR = (0, 0, 255)
LINES_CLOSED = True

CHANNELS = 3

'''
Returns a list of lines represented as lists of (x, y) coordinates.

page: the XML page element to search for lines in.
namespace: the namespace used in the XML file.
'''
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

'''
Produce a mask for the top and bottom lines of the text line bounding boxes.

gt_lines: the coordinates of the ground truth lines.
shape: the shape of the image
thickness: the thickness of the lines in the mask.
'''
def get_mask(gt_lines, shape, thickness):
    image_shape = (shape[0], shape[1], CHANNELS)
    top_line_image = np.zeros(image_shape, dtype=np.uint8)
    bottom_line_image = np.zeros(image_shape, dtype=np.uint8)

    # Draw each ground truth bounding box on image
    for line_coords in gt_lines:
        line_coords = np.array(line_coords, np.int32)
        centroid = np.mean(line_coords, axis=0, dtype=np.uint32) 
        first_quad = []
        second_quad = []
        third_quad = []
        fourth_quad = []

        # Group coordinates by quadrant
        for line_coord in line_coords:
            if line_coord[0] >= centroid[0]:
                if line_coord[1] <= centroid[1]:
                    first_quad.append(line_coord)
                else:
                    fourth_quad.append(line_coord)
            else:
                if line_coord[1] <= centroid[1]:
                    second_quad.append(line_coord)
                else:
                    third_quad.append(line_coord)

        # Compute distances from centroid for each coordinate
        first_quad_distances = np.array(list(map(np.linalg.norm, 
            first_quad - centroid)))
        second_quad_distances = np.array(list(map(np.linalg.norm, 
            second_quad - centroid)))
        third_quad_distances = np.array(list(map(np.linalg.norm, 
            third_quad - centroid)))
        fourth_quad_distances = np.array(list(map(np.linalg.norm, 
            fourth_quad - centroid)))

        # Find corner as the furthest point in each quadrant
        top_right_corner = first_quad[np.argmax(first_quad_distances)]
        top_left_corner = second_quad[np.argmax(second_quad_distances)]
        bottom_left_corner = third_quad[np.argmax(third_quad_distances)]
        bottom_right_corner = fourth_quad[np.argmax(fourth_quad_distances)]

        center_to_top_left = top_left_corner - centroid
        center_to_top_right = top_right_corner - centroid
        center_to_bottom_right = bottom_right_corner - centroid
        center_to_bottom_left = bottom_left_corner - centroid

        # Calculate range of angles with centroid as origin for each side, 
        # bounded by angles of corners
        top_line_angle_range = [atan2(-center_to_top_right[1], 
            center_to_top_right[0]), atan2(-center_to_top_left[1], 
                center_to_top_left[0])]
        right_line_angle_range = [atan2(-center_to_bottom_right[1], 
            center_to_bottom_right[0]), atan2(-center_to_top_right[1], 
                center_to_top_right[0])]
        bottom_line_angle_range = [atan2(-center_to_bottom_left[1], 
            center_to_bottom_left[0]), atan2(-center_to_bottom_right[1], 
                center_to_bottom_right[0])]

        # Create line groups for coordinates initialized with corners
        top_line = [[top_left_corner[0], top_left_corner[1]], 
                [top_right_corner[0], top_right_corner[1]]]
        right_line = [[top_right_corner[0], top_right_corner[1]], 
                [bottom_right_corner[0], bottom_right_corner[1]]]
        bottom_line = [[bottom_right_corner[0], bottom_right_corner[1]], 
                [bottom_left_corner[0], bottom_left_corner[1]]]
        left_line = [[bottom_left_corner[0], bottom_left_corner[1]], 
                [top_left_corner[0], top_left_corner[1]]]

        # Add each coordinate to the correct group depending on which range the
        # angle from the centroid origin falls in
        for line_coord in line_coords:
            angle = atan2(-(line_coord - centroid)[1], 
                    (line_coord - centroid)[0])

            # Skip if coordinate is a corner
            if (line_coord == top_left_corner).all() or \
                    (line_coord == top_right_corner).all() or \
                    (line_coord == bottom_right_corner).all() or \
                    (line_coord == bottom_left_corner).all():
                continue

            if angle > top_line_angle_range[0] and \
                    angle < top_line_angle_range[1]:
                top_line.append([line_coord[0], line_coord[1]])
            
            if angle > right_line_angle_range[0] and \
                    angle < right_line_angle_range[1]:
                right_line.append([line_coord[0], line_coord[1]])
            
            if angle > bottom_line_angle_range[0] and \
                    angle < bottom_line_angle_range[1]:
                bottom_line.append([line_coord[0], line_coord[1]])
            
            if angle < bottom_line_angle_range[0] or \
                    angle > top_line_angle_range[1]:
                left_line.append([line_coord[0], line_coord[1]])

        top_line = sorted(top_line, key=lambda k: [k[0], k[1]])
        bottom_line = sorted(bottom_line, key=lambda k: [k[0], k[1]])

        top_line_image = cv2.polylines(top_line_image, 
                [np.array(top_line, np.int32)], not LINES_CLOSED, TOP_LINE_COLOR, 
                thickness=thickness)
        bottom_line_image = cv2.polylines(bottom_line_image, 
                [np.array(bottom_line, np.int32)], not LINES_CLOSED, 
                BOTTOM_LINE_COLOR, thickness=thickness)

    overlap = cv2.bitwise_or(top_line_image, bottom_line_image)
    overlap = cv2.cvtColor(overlap, cv2.COLOR_RGB2BGR)

    return overlap

'''
Run the labeling tool with the given arguments.

argv: command line arguments.
'''
def main(argv):
    parser = argparse.ArgumentParser(description="A tool to produce masks for \
            document images. Currently only supports top and bottom lines of \
            text line bounding boxes.")
    parser.add_argument("gt_dir", help="The ground truth directory.")
    parser.add_argument("out_dir", help="The desired output directory.")
    parser.add_argument("-t", "--thickness", help="Line thickness for the \
            output mask.", default=THICKNESS, type=int, choices=range(1, THICKNESS + 1))
    args = parser.parse_args(argv)
    gt_dir = args.gt_dir
    out_dir = args.out_dir
    thickness = args.thickness

    for filename in os.listdir(gt_dir):
        print("Labeling " + filename + "...")

        gt_file = open(gt_dir + "/" + filename, 'r')

        gt_page = et.parse(gt_file).getroot().find(PAGE_TAG, NAMESPACE_GT)
        gt_lines = get_line_coords(gt_page, NAMESPACE_GT)
        gt_image_height = int(gt_page.get(HEIGHT_TAG))
        gt_image_width = int(gt_page.get(WIDTH_TAG))
        gt_image_shape = (gt_image_height, gt_image_width)

        cv2.imwrite(out_dir + "/" + filename.replace(".xml", ".png"), 
                get_mask(gt_lines, gt_image_shape, thickness))

        gt_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
