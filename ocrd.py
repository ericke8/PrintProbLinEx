import os
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

LOWER_BINARY_THRESH = 127
UPPER_BINARY_THRESH = 255

COORD_DELIM = ","

THICKNESS = 3
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
        #image = cv2.polylines(image, [line_coords], True, (0, 255, 0), 2)
        #continue
        centroid = np.mean(line_coords, axis=0, dtype=np.uint32) 
        first_quad = []
        second_quad = []
        third_quad = []
        fourth_quad = []

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

        first_quad_distances = np.array(list(map(np.linalg.norm, centroid - first_quad)))
        second_quad_distances = np.array(list(map(np.linalg.norm, centroid - second_quad)))
        third_quad_distances = np.array(list(map(np.linalg.norm, centroid - third_quad)))
        fourth_quad_distances = np.array(list(map(np.linalg.norm, centroid - fourth_quad)))

        top_right_corner = first_quad[np.argmax(first_quad_distances)]
        top_left_corner = second_quad[np.argmax(second_quad_distances)]
        bottom_left_corner = third_quad[np.argmax(third_quad_distances)]
        bottom_right_corner = fourth_quad[np.argmax(fourth_quad_distances)]

        top_line_angle_range = [atan2(-(top_right_corner - centroid)[1], (top_right_corner - centroid)[0]),
                atan2(-(top_left_corner - centroid)[1], (top_left_corner - centroid)[0])]
        right_line_angle_range = [atan2(-(bottom_right_corner - centroid)[1], (bottom_right_corner - centroid)[0]),
                atan2(-(top_right_corner - centroid)[1], (top_right_corner - centroid)[0])]
        bottom_line_angle_range = [atan2(-(bottom_left_corner - centroid)[1], (bottom_left_corner - centroid)[0]),
                atan2(-(bottom_right_corner - centroid)[1], (bottom_right_corner - centroid)[0])]


        top_line = [[top_left_corner[0], top_left_corner[1]], [top_right_corner[0], top_right_corner[1]]]
        right_line = [[top_right_corner[0], top_right_corner[1]], [bottom_right_corner[0], bottom_right_corner[1]]]
        bottom_line = [[bottom_right_corner[0], bottom_right_corner[1]], [bottom_left_corner[0], bottom_left_corner[1]]]
        left_line = [[bottom_left_corner[0], bottom_left_corner[1]], [top_left_corner[0], top_left_corner[1]]]

        for line_coord in line_coords:
            angle = atan2(-(line_coord - centroid)[1], (line_coord - centroid)[0])

            if (line_coord == top_left_corner).all() or (line_coord == top_right_corner).all() or \
                    (line_coord == bottom_right_corner).all() or (line_coord == bottom_left_corner).all():
                continue

            if angle >= top_line_angle_range[0] and angle <= top_line_angle_range[1]:
                top_line.append([line_coord[0], line_coord[1]])
            
            if angle >= right_line_angle_range[0] and angle <= right_line_angle_range[1]:
                right_line.append([line_coord[0], line_coord[1]])
            
            if angle >= bottom_line_angle_range[0] and angle <= bottom_line_angle_range[1]:
                bottom_line.append([line_coord[0], line_coord[1]])
            
            if angle <= bottom_line_angle_range[0] or angle >= top_line_angle_range[1]:
                left_line.append([line_coord[0], line_coord[1]])

        top_line = sorted(top_line, key=lambda k: [k[0], k[1]])
        #right_line = sorted(right_line, key=lambda k: [k[1], k[0]])
        bottom_line = sorted(bottom_line, key=lambda k: [k[0], k[1]])
        #left_line = sorted(left_line, key=lambda k: [k[1], k[0]])

        if len(line_coords) == 4:
            cv2.line(image, (top_left_corner[0], top_left_corner[1]), (top_right_corner[0], top_right_corner[1]), (255, 0, 0), THICKNESS)
            #cv2.line(image, (top_right_corner[0], top_right_corner[1]), (bottom_right_corner[0], bottom_right_corner[1]), (0, 255, 0), THICKNESS)
            cv2.line(image, (bottom_right_corner[0], bottom_right_corner[1]), (bottom_left_corner[0], bottom_left_corner[1]), (0, 0, 255), THICKNESS)
            #cv2.line(image, (bottom_left_corner[0], bottom_left_corner[1]), (top_left_corner[0], top_left_corner[1]), (0, 255, 255), THICKNESS)
        else:
            image = cv2.polylines(image, [np.array(top_line, np.int32)], not LINES_CLOSED, (255, 0, 0), thickness=THICKNESS)
            #image = cv2.polylines(image, [np.array(right_line, np.int32)], not LINES_CLOSED, (0, 255, 0), thickness=THICKNESS)
            image = cv2.polylines(image, [np.array(bottom_line, np.int32)], not LINES_CLOSED, (0, 0, 255), thickness=THICKNESS)
            #image = cv2.polylines(image, [np.array(left_line, np.int32)], not LINES_CLOSED, (0, 255, 255), thickness=THICKNESS)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

for filename in os.listdir("OCR-D_IMG"):
    basename = filename.replace(".png", "")
    print(basename)
    gt_file = open("OCR-D_GT/" + basename + ".xml", 'r')

    gt_page = et.parse(gt_file).getroot().find(PAGE_TAG, NAMESPACE_GT)
    gt_image_width = int(gt_page.get(WIDTH_TAG))
    gt_image_height = int(gt_page.get(HEIGHT_TAG))
    shape = (gt_image_height, gt_image_width)

    gt_lines = get_line_coords(gt_page, NAMESPACE_GT)

    image = cv2.imread("OCR-D_IMG/" + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imwrite("OCR-D_LABELS/" + filename, output_image(image, gt_lines))

    gt_file.close()
