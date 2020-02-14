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
Email: jasonvega14@yahoo.com
'''

import cv2
import matplotlib.pyplot as plt
import os
import sys
import getopt
import xml.etree.ElementTree as et
import numpy as np

MATCHSCORE_THRESHOLD = 0.95

OUTPUT_DIR_OPT = "-o"
IMAGE_DIR_OPT = "-i"
MATCH_THRESH_OPT = "-t"
LOWER_THRESH_OPT = "-l"
UPPER_THRESH_OPT = "-u"
THRESH_STEPS_OPT = "-s"
OPTIONS = "o:i:t:l:u:s:"
PRED_DIR_OPT = 0
GT_DIR_OPT = 1
ARGS = 2

DATA_EXTENSION = ".xml"
IMAGE_EXTENSION = ".tif"
RESULTS_FILE = "results"
RESULTS_RANGE_FILE = "results_range"
RESULTS_FILE_EXT = ".csv"

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

#THRESHOLD_STEPS = 0.01
THRESHOLD_STEPS = 100

GT_LINE_COLOR = (0, 255, 0)
PRED_LINE_COLOR = (255, 0, 0)
LINES_CLOSED = True

DETECTION_RESULT_IDX = 0
RECOGNITION_RESULT_IDX = 1
F_MEASURE_RESULT_IDX = 2

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
threshold: the threshold for one-to-one match acceptance.
'''
def get_one_to_one_matches(pred_lines, gt_lines, shape, threshold):
    matches = 0

    gt_lines = gt_lines.copy()

    for pred_line in pred_lines:
        pred_img = cv2.fillConvexPoly(np.zeros(shape, np.uint8), 
                                  np.array(pred_line, dtype=np.int32), 1)

        for gt_line in gt_lines:
            # Draw filled bounding boxes
            gt_img = cv2.fillConvexPoly(np.zeros(shape, np.uint8), 
                                        np.array(gt_line, dtype=np.int32), 1)

            matchscore = get_intersection_over_union(pred_img, gt_img)

            if matchscore >= threshold:
                matches += 1
                gt_lines.remove(gt_line)
                break

    return matches

'''
Returns the detection accuracy, recognition accuracy and F-measure for a given 
document.

pred_lines: a list of coordinates for predicted lines.
gt_lines: a list of coordinates for ground truth lines.
image: the document image.
threshold: the threshold for one-to-one match acceptance.
'''
def evaluate(pred_lines, gt_lines, image, threshold):
    pred_size = len(pred_lines)
    gt_size = len(gt_lines)
    matches = get_one_to_one_matches(pred_lines, gt_lines, image, threshold)
    detection_accuracy = 1 if not gt_size else (matches / gt_size)
    recognition_accuracy = 1 if not pred_size else (matches / pred_size)
    f_measure = 0
    
    if not pred_size and not gt_size:
        f_measure = 1
    else:
        f_measure = 2 * matches / (pred_size + gt_size)
    
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
Parse command line arguments. Returns a dictionary of the parsed arguments.

argv: command line arguments.
'''
def parse_args(argv):
    parsed_args = {}

    try:
        opts, args = getopt.getopt(argv, OPTIONS)

        if len(args) != ARGS:
            raise getopt.GetoptError("Invalid number of arguments specified.")

        for opt, arg in opts:
            if opt == OUTPUT_DIR_OPT:
                parsed_args[OUTPUT_DIR_OPT] = arg + "/" if arg[-1] != "/" else \
                        arg
            elif opt == IMAGE_DIR_OPT:
                parsed_args[IMAGE_DIR_OPT] = arg + "/" if arg[-1] != "/" else \
                        arg
            elif opt == MATCH_THRESH_OPT or opt == LOWER_THRESH_OPT or \
                    opt == UPPER_THRESH_OPT:
                arg = float(arg)

                if arg >= 0 and arg <= 1:
                    if opt == MATCH_THRESH_OPT:
                        parsed_args[MATCH_THRESH_OPT] = arg
                    elif opt == LOWER_THRESH_OPT:
                        parsed_args[LOWER_THRESH_OPT] = arg
                    elif opt == UPPER_THRESH_OPT:
                        parsed_args[UPPER_THRESH_OPT] = arg
                else:
                    raise getopt.GetoptError("Matchscore threshold must be " + \
                            "a real value between 0 and 1 (inclusive).")
            elif opt == THRESH_STEPS_OPT:
                arg = int(arg)

                if arg >= 0:
                    parsed_args[THRESH_STEPS_OPT] = arg
                else:
                    raise getopt.GetoptError("The number of steps should " + \
                            "be non-negative.")

        if parsed_args.get(MATCH_THRESH_OPT) and \
                (parsed_args.get(LOWER_THRESH_OPT) or \
                parsed_args.get(UPPER_THRESH_OPT)):
            raise getopt.GetoptError("Cannot simultaneously set threshold " + \
                    "value while specifying a threshold range.")

        if parsed_args.get(LOWER_THRESH_OPT) and \
                parsed_args.get(UPPER_THRESH_OPT) and \
                parsed_args.get(LOWER_THRESH_OPT) > \
                parsed_args.get(UPPER_THRESH_OPT):
            raise getopt.GetoptError("The lower threshold should be less " + \
                    "than or equal to the upper threshold.")


        if parsed_args.get(IMAGE_DIR_OPT) and not parsed_args.get(OUTPUT_DIR_OPT):
            raise getopt.GetoptError("-o flag needs to be specfied when -i " + \
                    "flag is used.")

        parsed_args[PRED_DIR_OPT] = args[PRED_DIR_OPT] + "/" if \
                args[PRED_DIR_OPT][-1] != "/" else args[PRED_DIR_OPT]
        parsed_args[GT_DIR_OPT] = args[GT_DIR_OPT] + "/" if \
                args[GT_DIR_OPT][-1] != "/" else args[GT_DIR_OPT]
    except getopt.GetoptError as err:
        print("ERROR: " + str(err) + "\n")
        print("Usage: python page_xml_evaluator.py [OPTION]... pred_dir " + \
                "gt_dir\n")
        print("pred_dir - the directory containing the XML files to evaluate.")
        print("gt_dir - the directory containing the ground truth XML " + \
                "files.\n")
        print("Optional flags:")
        print("-o out_dir - output evaluation results and document images " + \
                "with ground truth and prediction overlay to the desired " + \
                "output directory (out_dir).")
        print("-i image_dir - include document images with ground truth " + \
                "and prediction overlay in output. The -o flag must also " + \
                "be set. image_dir is the directory containing the " + \
                "document images.")
        print("-t n - sets the matchscore threshold. Must be a real value " + \
                "between 0 and 1 (inclusive). Cannot be set alongside -u " + \
                "or -l. Default value is " + str(MATCHSCORE_THRESHOLD) + ".")
        print("-l n - sets the lower threshold to be n. A range of " + \
                "thresholds will then be used for evaluation. Must be a " + \
                "real between 0 and 1 (inclusive). Cannot be set alongside " + \
                "-t. Default value is 0 if only -u is set.")
        print("-u n - sets the upper threshold to be n. A range of " + \
                "will then be used for evaluation. Must be a real between " + \
                "0 and 1 (inclusive). Cannot be set alongside -t. " + \
                "Default value is 1 if only -l is set.")
        print("-s n - the number of steps when evaluating over a range of " + \
                "threshold values. Default value is " + str(THRESHOLD_STEPS) + \
                ".")

        sys.exit()

    return parsed_args


'''
Run evaluation tool with the given command line arguments.

argv: command line arguments.
'''
def main(argv):
    args = parse_args(argv)

    image_dir = args.get(IMAGE_DIR_OPT, "")
    pred_dir = args.get(PRED_DIR_OPT, "")
    gt_dir = args.get(GT_DIR_OPT, "")
    out_dir = args.get(OUTPUT_DIR_OPT, "")

    iterate_thresh = args.get(LOWER_THRESH_OPT) or args.get(UPPER_THRESH_OPT)
    lower_threshold = args.get(LOWER_THRESH_OPT, 0)
    upper_threshold = args.get(UPPER_THRESH_OPT, 1)
    matchscore_threshold = (lower_threshold if iterate_thresh else \
            args.get(MATCH_THRESH_OPT, MATCHSCORE_THRESHOLD))
    threshold_steps = args.get(THRESH_STEPS_OPT, THRESHOLD_STEPS)

    images_outputted = False

    results_range_output = out_dir != "" and iterate_thresh
    results_range_file = None
    results_range_file_name = RESULTS_RANGE_FILE + RESULTS_FILE_EXT

    try: 
        file_names = os.listdir(pred_dir)

        if file_names:
            file_names.sort()
    except FileNotFoundError:
        print("Directory " + pred_dir + " not found. Aborting evaluation.")
        sys.exit()

    if results_range_output:
        try:
            results_range_file = open(out_dir + results_range_file_name, 'w+')
            results_range_file.write("Threshold,Average Detection " + \
                    "Accuracy, Average Recognition Accuracy, Average " + \
                    "F-measure\n")
        except FileNotFoundError:
            print("Could not open " + out_dir + results_range_file_name + 
                    " for writing. Check that the specified output " + \
                    "directory exists.\n")
            results_range_output = False

    #Iterate through all matchscore thresholds if range is specified
    for matchscore_threshold in np.linspace(matchscore_threshold, \
            upper_threshold, threshold_steps):
        results_file = None
        results_file_name = RESULTS_FILE + "_" + \
                str(matchscore_threshold).replace("0.", "") + RESULTS_FILE_EXT
        results_output = out_dir != ""
        
        detection_accuracy = 0
        recognition_accuracy = 0
        f_measure = 0
        evaluated = 0

        skipped_evals = []
        image_outputted = False

        if out_dir:
            try:
                results_file = open(out_dir + results_file_name, 'w+')
                results_file.write("File,Detection Accuracy,Recognition " + \
                        "Accuracy,F-measure\n")
            except FileNotFoundError:
                print("Could not open " + out_dir + results_file_name + \
                        "for writing. Check that the specified output " + \
                        "directory exists.\n")
                results_output = False

        print("--- Beginning evaluation on " + pred_dir + " with threshold " + \
                str(matchscore_threshold) + " ---\n")

        for pred_file in file_names:
            image_filename = pred_file.replace(DATA_EXTENSION, IMAGE_EXTENSION)
            gt_filename = pred_file.replace(IMAGE_EXTENSION, DATA_EXTENSION)

            image = None
            pred_page = None
            gt_page = None

            if not images_outputted and out_dir and image_dir:
                try:
                    image = cv2.imread(image_dir + image_filename)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_output = True
                except cv2.error:
                    print("Image file " + image_dir + image_filename + 
                            " could not be read. Skipping image output.\n")

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

            # Write output image
            if not images_outputted and out_dir and image_output and image.size:
                try:
                    cv2.imwrite(out_dir + image_filename, output_image(image, 
                        pred_lines, gt_lines))
                except cv2.error:
                    print("Image file " + out_dir + image_filename + \
                            " could not be wrriten to. Skipping image " + \
                            "output. \n")

            result = evaluate(pred_lines, gt_lines, 
                    (gt_image_height, gt_image_width), matchscore_threshold)
            detection_accuracy += result[DETECTION_RESULT_IDX]
            recognition_accuracy += result[RECOGNITION_RESULT_IDX]
            f_measure += result[F_MEASURE_RESULT_IDX]
            evaluated += 1

            if results_output:
                results_file.write(pred_file + "," + \
                        str(result[DETECTION_RESULT_IDX]) + "," + \
                        str(result[RECOGNITION_RESULT_IDX]) + "," + \
                        str(result[F_MEASURE_RESULT_IDX]) + "\n")

            print("DA: " + str(result[DETECTION_RESULT_IDX]) + ", RA: " + \
                    str(result[RECOGNITION_RESULT_IDX]) + ", F: " + \
                    str(result[F_MEASURE_RESULT_IDX]) + "\n")

        if evaluated:
            detection_accuracy /= evaluated
            recognition_accuracy /= evaluated
            f_measure /= evaluated

            print("--- Global Results ---")
            print("DA: " + str(detection_accuracy) + 
                    ", RA: " + str(recognition_accuracy) + ", F: " + \
                            str(f_measure))

            if results_range_output:
                results_range_file.write(str(matchscore_threshold) + "," + \
                        str(detection_accuracy) + "," + \
                        str(recognition_accuracy) + "," + str(f_measure) + "\n")

            if len(skipped_evals) > 0:
                print("\nSkipped evaluations (" + str(len(skipped_evals)) + \
                        "):")

                for file_name in skipped_evals:
                    print(file_name)
        
                print()
        else:
            print("No files evaluated. Check that the ground truth " + \
                    "directory exists and contains valid files.")

        if results_output:
            results_file.close()

        # Ensure images are only outputted on the first iteration
        if not images_outputted:
            images_outputted = True
    
        # End iteration if range was not specified
        if not iterate_thresh:
            break

    if results_range_output:
        results_range_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
