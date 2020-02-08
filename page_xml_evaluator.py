'''
Line segmentation evaluation tool for PAGE XML data. Evaluation metrics are 
detection accuracy, recognition accuracy and F-measure, all based on the 
intersection over union (IU) score.

For each prediction file being evaluated, the corresponding image file and 
ground truth file must have the same name as the prediction_file. Only 
the prediction XML files should exist in their directory. The imageWidth 
and imageHeight attributes on the Page element should be the same for the
predicted and corresponding ground truth XML file.

NOTE: NAMESPACE_PRED and NAMESPACE_GT may need to be modified if XML data for
predictions and/or ground truth use different schemas.

Author: Jason Vega
'''

import cv2
import matplotlib.pyplot as plt
import os
import sys
import getopt
import xml.etree.ElementTree as et
import numpy as np

MATCHSCORE_THRESHOLD = 0.8

OUTPUT_DIR_OPT = "-o"
IMAGE_DIR_OPT = "-i"
OPTIONS = "o:i:"
PRED_DIR_OPT = 0
GT_DIR_OPT = 1
ARGS = 2

DATA_EXTENSION = ".xml"
IMAGE_EXTENSION = ".tif"
RESULTS_FILE = "results.csv"

PAGE_TAG = "ns0:Page"
TEXTREGION_TAG = "ns0:TextRegion"
TEXTLINE_TAG = "ns0:TextLine"
COORDS_TAG = "ns0:Coords"
POINTS_ATTR = "points"
WIDTH_TAG = "imageWidth"
HEIGHT_TAG = "imageHeight"

COORD_DELIM = ","

LOWER_BINARY_THRESH = 127
UPPER_BINARY_THRESH = 255

PRED_LINE_COLOR = (255, 0, 0)
GT_LINE_COLOR = (0, 255, 0)
LINES_CLOSED = True
FIGURE_SIZE = 35

NAMESPACE_PRED = {
    "ns0": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

NAMESPACE_GT = {
    "ns0": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

'''
Returns a list of the text line coordinates for the entire page.

page: the page element to search for text line coordinates.
namespace: namespace for this page.
'''
def get_line_coords(page, namespace):
    lineList = []
    
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
            
            lineList.append(coords)
                
    return lineList

'''
Returns the intersection over union (IU) value of the two bounding boxes.

first: an image containing the first (filled) bounding box.
second: an image containing the second (filled) bounding box.
'''
def get_intersection_over_union(first, second):
    intersection = cv2.bitwise_and(first, second)
    intersection_score = cv2.countNonZero(intersection)

    union = cv2.bitwise_or(first, second)
    union_score = cv2.countNonZero(union)

    return intersection_score / union_score

'''
Returns the number of one-to-one matches between the predicted lines and ground 
truth. A one-to-one match occurs when the matchscore between two lines is 
greater than the specified threshold.

pred_lines: a list of coordinates for predicted lines.
gt_lines: a list of coordinates for ground truth lines.
shape: shape of the document image.
'''
def get_one_to_one_matches(pred_lines, gt_lines, shape):
    matches = 0
    
    for pred_line in pred_lines:
        for gt_line in gt_lines:
            # Draw filled bounding boxes
            pred_img = cv2.fillConvexPoly(np.zeros(shape, np.uint8), 
                                          np.array(pred_line, dtype=np.int32), 1)
            gt_img = cv2.fillConvexPoly(np.zeros(shape, np.uint8), 
                                        np.array(gt_line, dtype=np.int32), 1)

            matchscore = get_intersection_over_union(pred_img, gt_img)

            if matchscore > MATCHSCORE_THRESHOLD:
                matches += 1

    return matches

'''
Returns the detection accuracy, recognition accuracy and F-measure for a given 
document.

pred_lines: a list of coordinates for predicted lines.
gt_lines: a list of coordinates for ground truth lines.
image: the document image.
'''
def evaluate(pred_lines, gt_lines, image):
    matches = get_one_to_one_matches(pred_lines, gt_lines, image)
    detection_accuracy = 1 if not len(gt_lines) else (matches / len(gt_lines))
    recognition_accuracy = 1 if not len(pred_lines) \
            else (matches / len(pred_lines))
    f_measure = 0
    
    if not len(pred_lines) and not len(gt_lines):
        f_measure = 1
    else:
        f_measure = 2 * matches / (len(pred_lines) + len(gt_lines))
    
    return (detection_accuracy, recognition_accuracy, f_measure)

'''
Returns the document image with optional overlayed predicted line extractions 
and ground truth.

image: the document image.
pred_lines: a list of coordinates for predicted lines. These will be displayed 
in green.
gt_lines: a list of coordinates for ground truth lines. These will be displayed 
in red.
'''
def output_image(image, pred_lines, gt_lines):
    if pred_lines:
        # Draw each predicted line bounding box on image
        for lineCoords in pred_lines:
            lineCoords = np.array(lineCoords, np.int32)
            image = cv2.polylines(image, [lineCoords], LINES_CLOSED, 
                    PRED_LINE_COLOR)
                
    if gt_lines:
        # Draw each ground truth bounding box on image
        for lineCoords in gt_lines:
            lineCoords = np.array(lineCoords, np.int32)
            image = cv2.polylines(image, [lineCoords], LINES_CLOSED, 
                    GT_LINE_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

'''
Process command line arguments and run evaluations.

argv: command line arguments.
'''
def main(argv):
    image_dir = ""
    pred_dir = ""
    gt_dir = ""
    out_dir = ""

    # Parse command line args
    try:
        opts, args = getopt.getopt(argv, OPTIONS)

        if len(args) != ARGS:
            raise getopt.GetoptError("Invalid number of arguments specified.")

        for opt, arg in opts:
            if opt == OUTPUT_DIR_OPT:
                out_dir = arg + "/" if arg[-1] != "/" else arg
            elif opt == IMAGE_DIR_OPT:
                image_dir = arg + "/" if arg[-1] != "/" else arg
            else:
                raise getopt.GetoptError("Unrecognized option passed.")


        if image_dir and not out_dir:
            raise getopt.GetoptError("-o flag needs to be specfied when -i " + \
                    "flag is used")

        pred_dir = args[PRED_DIR_OPT] + "/" if args[PRED_DIR_OPT][-1] != "/" \
                else args[PRED_DIR_OPT]
        gt_dir = args[GT_DIR_OPT] + "/" if args[GT_DIR_OPT][-1] != "/" \
                else args[GT_DIR_OPT]
    except getopt.GetoptError:
        print("Usage: python3 page_xml_evaluator.py -o out_dir -i " + \
                "image_dir pred_dir gt_dir\n")
        print("pred_dir - the directory containing the XML files to evaluate.")
        print("gt_dir - the directory containing the ground truth XML " + \
                "files.\n")
        print("Optional flags:\n")
        print("-o out_dir - output evaluation results and document images " + \
                "with ground truth and prediction overlay to the desired " + \
                "output directory (out_dir).")
        print("-i image_dir - include document images with ground truth " + \
                "and prediction overlay in output. The -o flag must also " + \
                "be set. image_dir is the directory containing the " + \
                "document images.")

        sys.exit()

    results_file = None
    results_output = out_dir != ""

    if out_dir:
        try:
            results_file = open(out_dir + RESULTS_FILE, 'w+')
            results_file.write("File,Detection Accuracy,Recognition " + \
                    "Accuracy,F-measure\n")
        except FileNotFoundError:
            print("Could not open " + out_dir + RESULTS_FILE + " for " + \
                    "writing. Check that the specified output directory " + \
                    "exists.\n")
            results_output = False

    skipped_evals = []
    detection_accuracy = 0
    recognition_accuracy = 0
    f_measure = 0
    evaluated = 0
    image = None

    try: 
        file_names = os.listdir(pred_dir)

        if file_names:
            file_names.sort()

        for pred_file in file_names:
            image_filename = pred_file.replace(DATA_EXTENSION, IMAGE_EXTENSION)
            gt_filename = pred_file.replace(IMAGE_EXTENSION, DATA_EXTENSION)
            image_output = True

            if out_dir and image_dir:
                try:
                    image = cv2.imread(image_dir + image_filename)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except cv2.error:
                    print("Image file " + image_dir + image_filename + 
                            " could not be read. Skipping image output.\n")
                    image_output = False

            pred_page = None
            
            # Skip evaluation if prediction file cannot be opened or parsed.
            try:
                pred_page = et.parse(pred_dir + pred_file).getroot() \
                    .find(PAGE_TAG, NAMESPACE_PRED)
            except IOError:
                print("Could not open " + pred_dir + pred_file + \
                        ". Skipping evaluation.\n")
                skipped_evals.append(pred_file)
                continue
            except et.ParseError:
                print("Could not parse " + pred_dir + pred_file + \
                        ". Skipping evaluation.\n")
                skipped_evals.append(pred_file)
                continue

            gt_page = None

            # Skip evaluation if ground truth files cannot be parsed or opened.
            try:
                gt_page = et.parse(gt_dir + \
                        gt_filename).getroot().find(PAGE_TAG, NAMESPACE_GT)
            except IOError:
                print("Could not open " + gt_dir + gt_filename + \
                        ". Skipping evaluation.\n")
                skipped_evals.append(pred_file)
                continue
            except et.ParseError:
                print("Could not parse " + gt_dir + gt_filename + \
                        ". Skipping evaluation.\n")
                skipped_evals.append(pred_file)
                continue

            gt_image_width = int(gt_page.get(WIDTH_TAG))
            gt_image_height = int(gt_page.get(HEIGHT_TAG))
            pred_image_width = int(pred_page.get(WIDTH_TAG))
            pred_image_height = int(pred_page.get(HEIGHT_TAG))

            # Skip evaluation if image dimensions do not match
            if gt_image_width != pred_image_width or \
                gt_image_height != pred_image_height:
                print("Ground truth and prediction image dimensions do " + \
                        "not match. Skipping evaluation. \n")
                skipped_evals.append(pred_file)
                continue

            print("Evaluating " + pred_file + "...")

            pred_lines = get_line_coords(pred_page, NAMESPACE_PRED)
            gt_lines = get_line_coords(gt_page, NAMESPACE_GT)

            result = evaluate(pred_lines, gt_lines, 
                    (gt_image_height, gt_image_width))
            detection_accuracy += result[0]
            recognition_accuracy += result[1]
            f_measure += result[2]
            evaluated += 1

            if out_dir and image_output and image.size:
                try:
                    cv2.imwrite(out_dir + image_filename, output_image(image, 
                        pred_lines, gt_lines))
                except cv2.error:
                    print("Image file " + out_dir + image_filename + \
                            " could not be wrriten to. Skipping image " + \
                            "output. \n")

            if results_output:
                results_file.write(pred_file + "," + str(result[0]) + "," + 
                        str(result[1]) + "," + str(result[2]) + "\n")

            print("DA: " + str(result[0]) + ", RA: " + str(result[1]) +  
                    ", F: " + str(result[2]) + "\n")
    except FileNotFoundError:
        print("Directory " + pred_dir + " not found. Aborting evaluation.")
        sys.exit()

    if evaluated:
        detection_accuracy /= evaluated
        recognition_accuracy /= evaluated
        f_measure /= evaluated
    else:
        print("\nNo files evaluated. Check that the ground truth directory " + \
            "exists and contains valid files.")
        sys.exit()

    print("\n--- Global Results ---\nDA: " + str(detection_accuracy) + 
            ", RA: " + str(recognition_accuracy) + ", F: " + str(f_measure))

    if len(skipped_evals) > 0:
        print("\nSkipped evaluations (" + str(len(skipped_evals)) + "):\n")

    for file_name in skipped_evals:
        print(file_name)
    
    if results_output:
        results_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
