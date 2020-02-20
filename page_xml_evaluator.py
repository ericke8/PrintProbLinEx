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
IMAGE_OUT_OPT = "-i"
MATCH_THRESH_OPT = "-t"
LOWER_THRESH_OPT = "-l"
UPPER_THRESH_OPT = "-u"
THRESH_STEPS_OPT = "-s"
IMAGE_OUT_PRED_ARG = "p"
IMAGE_OUT_GT_ARG = "g"
IMAGE_OUT_BOTH_ARG = "b"
OPTIONS = "o:i:t:l:u:s:"
IMAGE_DIR_OPT = 0
PRED_DIR_OPT = 1
GT_DIR_OPT = 2
ARGS = 3

DATA_EXTENSION = ".xml"
IMAGE_EXTENSION = ".tif"
RESULTS_FILE_EXT = ".csv"
RESULTS_FILE = "results"
RESULTS_RANGE_FILE = "results_range"
IU_FILE = "iu_scores"

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

THRESHOLD_STEPS = 101

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
image: the inverted binarized original image.
'''
def get_intersection_over_union(first, second, image):
    intersection = cv2.bitwise_and(cv2.bitwise_and(first, second), image)
    intersection_score = cv2.countNonZero(intersection)

    union = cv2.bitwise_and(cv2.bitwise_or(first, second), image)
    union_score = cv2.countNonZero(union)

    return intersection_score / union_score

'''
Returns a list of the maximum intersection-over-union score for each one-to-one
assignment. Each predicted line is assigned to a unique ground truth line that 
gives that maximum score. If there are no more ground truth lines to assign, 
the reset of the predicted lines will be skipped.

pred_lines: a list of coordinates for predicted lines.
gt_lines: a list of coordinates for ground truth lines.
shape: shape of the document image.
image: the inverted binarized original image.
'''
def get_maximum_iu_scores(pred_lines, gt_lines, shape, image):
    '''
    Finds the maximum IU scores by first iterating over set_1 and pruning set_2

    set_1: the set to iterate over.
    set_2: the set to prune.
    '''
    def find_maximum_iu_scores(set_1, set_2):
        iu_scores = []
        unmatched = 0

        set_2 = set_2.copy()

        for line_1 in set_1:
            if set_2:
                img_1 = cv2.fillConvexPoly(np.zeros(shape, np.uint8), 
                                          np.array(line_1, dtype=np.int32), 1)
                max_iu_score = 0
                max_iu_score_line = None

                for line_2 in set_2:
                    img_2 = cv2.fillConvexPoly(np.zeros(shape, np.uint8), 
                                                np.array(line_2, dtype=np.int32), 
                                                1)

                    matchscore = get_intersection_over_union(img_1, img_2, 
                            image)

                    # Make assignments if IU score is > 0 first
                    if matchscore > max_iu_score:
                        max_iu_score = matchscore
                        max_iu_score_line = line_2

                if max_iu_score_line:
                    iu_scores.append(max_iu_score)
                    set_2.remove(max_iu_score_line)
                else:
                    unmatched += 1
            else:
                unmatched += 1

        # Assign remaining lines arbitrarily
        iu_scores += min(unmatched, len(set_2)) * [0]

        return iu_scores
    
    # Compare two orders of iteration and prefer the higher sum of IU scores
    iu_scores_1 = find_maximum_iu_scores(pred_lines, gt_lines)
    iu_scores_2 = find_maximum_iu_scores(gt_lines, pred_lines)

    if sum(iu_scores_1) > sum(iu_scores_2):
        return iu_scores_1
    else:
        return iu_scores_2

'''
Returns the number of one-to-one matches between the predicted lines and ground 
truth. A one-to-one match is defined when the intersection-over-union score 
between a predicted line and a ground truth line meets the given threshold.

iu_scores: maximum intersection-over-union scores for all predicted lines, 
    where no two predicted lines are assigned the same ground truth line.
threshold: the threshold for one-to-one match acceptance.
'''
def get_one_to_one_matches(iu_scores, threshold):
    matches = 0

    for iu_score in iu_scores:
        if iu_score >= threshold:
            matches += 1

    return matches

'''
Returns the detection accuracy, recognition accuracy, and F-measure for a 
given document.

pred_size: the number of predicted lines.
gt_size: the number of ground truth lines.
iu_scores: maximum intersection-over-union scores for all predicted lines, 
    where no two predicted lines are assigned the same ground truth line.
threshold: the threshold for one-to-one match acceptance.
'''
def evaluate(pred_size, gt_size, iu_scores, threshold):
    matches = get_one_to_one_matches(iu_scores, threshold)
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
image_out_pred: whether or not to include prediction overlay on image output.
image_out_gt: whether or not to include ground truth overlay on image output.
'''
def output_image(image, pred_lines, gt_lines, image_out_pred, image_out_gt):
    if image_out_pred and pred_lines:
        # Draw each predicted line bounding box on image
        for lineCoords in pred_lines:
            lineCoords = np.array(lineCoords, np.int32)
            image = cv2.polylines(image, [lineCoords], LINES_CLOSED, 
                    PRED_LINE_COLOR)
                
    if image_out_gt and gt_lines:
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
            elif opt == IMAGE_OUT_OPT:
                if arg == IMAGE_OUT_PRED_ARG:
                    parsed_args[IMAGE_OUT_PRED_ARG] = True 
                elif arg == IMAGE_OUT_GT_ARG:
                    parsed_args[IMAGE_OUT_GT_ARG] = True
                elif arg == IMAGE_OUT_BOTH_ARG:
                    parsed_args[IMAGE_OUT_PRED_ARG] = True 
                    parsed_args[IMAGE_OUT_GT_ARG] = True
                else:
                    raise getopt.GetoptError("Invalid image output " + 
                        "argument specified.")
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


        if parsed_args.get(IMAGE_OUT_OPT) and not parsed_args.get(OUTPUT_DIR_OPT):
            raise getopt.GetoptError("-o flag needs to be specfied when -i " + \
                    "flag is used.")

        parsed_args[IMAGE_DIR_OPT] = args[IMAGE_DIR_OPT] + "/" if \
                args[IMAGE_DIR_OPT][-1] != "/" else args[IMAGE_DIR_OPT]
        parsed_args[PRED_DIR_OPT] = args[PRED_DIR_OPT] + "/" if \
                args[PRED_DIR_OPT][-1] != "/" else args[PRED_DIR_OPT]
        parsed_args[GT_DIR_OPT] = args[GT_DIR_OPT] + "/" if \
                args[GT_DIR_OPT][-1] != "/" else args[GT_DIR_OPT]
    except getopt.GetoptError as err:
        print("ERROR: " + str(err) + "\n")
        print("Usage: python page_xml_evaluator.py [OPTION]... image_dir " + \
                "pred_dir gt_dir\n")
        print("image_dir - the directory containing the original images.")
        print("pred_dir - the directory containing the XML files to evaluate.")
        print("gt_dir - the directory containing the ground truth XML " + \
                "files.\n")
        print("Optional flags:")
        print("-o out_dir - output evaluation results and document images " + \
                "with ground truth and prediction overlay to the desired " + \
                "output directory (out_dir).")
        print("-i  include document images with ground truth " + \
                "and prediction overlay in output. The -o flag must also " + \
                "be set.")
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
Retrieves predicted line coorindates, ground truth coordinates, image shapes, 
and maximum intersection-over-union scores for each one-to-one match in each 
file. Also outputs image files if desired by the user.

file_names: a list of the file names in pred_dir.
pred_dir: the directory containing prediction files.
gt_dir: the directory containing ground truth files.
our_dir: the output directory.
image_dir: the directory containg the original images.
image_out_pred: whether or not to include prediction overlay on image output.
image_out_gt: whether or not to include ground truth overlay on image output.
'''
def preprocess(file_names, pred_dir, gt_dir, out_dir, image_dir, image_out_pred, 
        image_out_gt):
    pred_lines = {}
    gt_lines = {}
    iu_scores = {}
    skipped_evals = []
    image_output = image_out_pred or image_out_gt
    iu_file = None
    iu_file_name = IU_FILE + RESULTS_FILE_EXT
    iu_file_output = out_dir != ""
    weighted_avg_iu_score = 0

    if out_dir:
        try:
            iu_file = open(out_dir + iu_file_name, 'w+')
            iu_file.write("File,Matches,Mean IU\n")
        except FileNotFoundError:
            print("Could not open " + out_dir + iu_file_name + 
                    " for writing. Check that the specified output " + \
                    "directory exists.\n")
            iu_file_output = False

    for pred_file in file_names:
        gt_filename = pred_file.replace(IMAGE_EXTENSION, DATA_EXTENSION)
        image_filename = pred_file.replace(DATA_EXTENSION, IMAGE_EXTENSION)
        pred_page = None
        gt_page = None
        image = None
        binarized = None

        print("Processing " + pred_file + "...")

        # Parse prediction file
        try:
            pred_page = et.parse(pred_dir + pred_file).getroot() \
                .find(PAGE_TAG, NAMESPACE_PRED)
        except IOError:
            print("Could not open " + pred_dir + pred_file + \
                    ". File will not be evaluated.\n")
            skipped_evals.append(pred_file)
            continue
        except et.ParseError:
            print("Could not parse " + pred_dir + pred_file + \
                    ". File will not be evaluated.\n")
            skipped_evals.append(pred_file)
            continue

        # Parse corresponding ground truth file
        try:
            gt_page = et.parse(gt_dir + \
                    gt_filename).getroot().find(PAGE_TAG, NAMESPACE_GT)
        except IOError:
            print("Could not open " + gt_dir + gt_filename + \
                    ". File will not be evaluated.\n")
            skipped_evals.append(pred_file)
            continue
        except et.ParseError:
            print("Could not parse " + gt_dir + gt_filename + \
                    ". File will not be evaluated.\n")
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
                    "not match. File will not be evaluated.\n")
            skipped_evals.append(pred_file)
            continue

        shape = (gt_image_height, gt_image_width)

        # Get prediction and ground truth line coordinates
        pred_lines[pred_file] = get_line_coords(pred_page, NAMESPACE_PRED)
        gt_lines[pred_file] = get_line_coords(gt_page, NAMESPACE_GT)

        # Retrieve corresponding original image
        try:
            image = cv2.imread(image_dir + image_filename)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ret, binarized = cv2.threshold(grayscale, LOWER_BINARY_THRESH, 
                    UPPER_BINARY_THRESH, cv2.THRESH_BINARY_INV)
        except cv2.error:
            print("Image file " + image_dir + image_filename + 
                    " could not be read. File will not be evaluated.")
            skipped_evals.append(pred_file)
            continue


        # Project predicted and ground truth bounding boxes onto original image 
        # and write this image to disk
        if out_dir and image_output and image.size:
            try:
                print("Writing image output to disk...")
                cv2.imwrite(out_dir + image_filename, output_image(image, 
                    pred_lines[pred_file], gt_lines[pred_file], 
                    image_out_pred, image_out_gt))
            except cv2.error:
                print("Image file " + out_dir + image_filename + \
                        " could not be wrriten to. Skipping image " + \
                        "output.")

        print("Calculating IU scores...")
        iu_scores[pred_file] = get_maximum_iu_scores(pred_lines[pred_file], 
                gt_lines[pred_file], shape, binarized)
        weighted_avg_iu_score += sum(iu_scores[pred_file])
        mean_iu_score = 0 if not iu_scores[pred_file] else \
                sum(iu_scores[pred_file]) / len(iu_scores[pred_file])
        print("Mean IU score: " + str(mean_iu_score))

        if out_dir and iu_file_output:
            iu_file.write(pred_file + "," + str(len(iu_scores[pred_file])) + \
                    "," + str(mean_iu_score) + "\n")

        print()

    weighted_avg_iu_score /= sum(len(iu_scores[filename]) for filename in \
            iu_scores)
    print("Global mean IU score: " + str(weighted_avg_iu_score) + "\n")

    # Print out files that will not be evaluated
    if len(skipped_evals) > 0:
        print("Files that will not be evaluated (" + \
                str(len(skipped_evals)) + "):")

        for file_name in skipped_evals:
            print(file_name)

        print()

    if iu_file_output:
        iu_file.close()

    return (pred_lines, gt_lines, iu_scores)

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

    image_out_pred = args.get(IMAGE_OUT_PRED_ARG, False)
    image_out_gt = args.get(IMAGE_OUT_GT_ARG, False)

    iterate_thresh = args.get(LOWER_THRESH_OPT) != None or \
            args.get(UPPER_THRESH_OPT) != None
    lower_threshold = args.get(LOWER_THRESH_OPT, 0)
    upper_threshold = args.get(UPPER_THRESH_OPT, 1)
    matchscore_threshold = (lower_threshold if iterate_thresh else \
            args.get(MATCH_THRESH_OPT, MATCHSCORE_THRESHOLD))
    threshold_steps = args.get(THRESH_STEPS_OPT, THRESHOLD_STEPS)

    results_range_output = out_dir != "" and iterate_thresh
    results_range_file = None
    results_range_file_name = RESULTS_RANGE_FILE + RESULTS_FILE_EXT

    pred_lines = {}
    gt_lines = {}
    iu_scores = {}

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
            results_range_file.write("Threshold,Detection " + \
                    "Accuracy,Recognition Accuracy,F-measure\n")
        except FileNotFoundError:
            print("Could not open " + out_dir + results_range_file_name + 
                    " for writing. Check that the specified output " + \
                    "directory exists.\n")
            results_range_output = False

    # Pre-process files for evaluation
    print("Pre-processing files for evaluation...\n")
    pred_lines, gt_lines, iu_scores = \
        preprocess(file_names, pred_dir, gt_dir, out_dir, image_dir, 
                image_out_pred, image_out_gt)

    #Iterate through all matchscore thresholds if range is specified
    for matchscore_threshold in np.linspace(matchscore_threshold, \
            upper_threshold, threshold_steps):
        results_file = None
        results_file_name = RESULTS_FILE + "_" + \
                str(matchscore_threshold).replace("0.", "") + RESULTS_FILE_EXT
        results_output = out_dir != ""
        
        detection_accuracy = 0
        recognition_accuracy = 0
        evaluated = 0

        if out_dir:
            try:
                results_file = open(out_dir + results_file_name, 'w+')
                results_file.write("File,Ground Truth Lines,Predicted " + \
                        "Lines,Detection Accuracy," + \
                        "Recognition Accuracy,F-measure\n")
            except FileNotFoundError:
                print("Could not open " + out_dir + results_file_name + \
                        "for writing. Check that the specified output " + \
                        "directory exists.\n")
                results_output = False

        print("--- Beginning evaluation on " + pred_dir + " with threshold " + \
                str(matchscore_threshold) + " ---\n")

        for pred_file in pred_lines:
            print("Evaluating " + pred_file + "...")

            num_gt_lines = len(gt_lines[pred_file])
            num_pred_lines = len(pred_lines[pred_file])
            result = evaluate(num_pred_lines, num_gt_lines, 
                    iu_scores[pred_file], matchscore_threshold)
            detection_accuracy += num_gt_lines * \
                    result[DETECTION_RESULT_IDX]
            recognition_accuracy += num_pred_lines * \
                    result[RECOGNITION_RESULT_IDX]
            evaluated += 1

            if results_output:
                results_file.write(pred_file + "," + \
                        str(len(gt_lines[pred_file])) + "," + \
                        str(len(pred_lines[pred_file])) + "," + \
                        str(result[DETECTION_RESULT_IDX]) + "," + \
                        str(result[RECOGNITION_RESULT_IDX]) + "," + \
                        str(result[F_MEASURE_RESULT_IDX]) + "\n")

            print("DA: " + str(result[DETECTION_RESULT_IDX]) + ", RA: " + \
                    str(result[RECOGNITION_RESULT_IDX]) + ", F: " + \
                    str(result[F_MEASURE_RESULT_IDX]) + "\n")

        if evaluated:
            total_gt_lines = sum(len(gt_lines[file_name]) for file_name in \
                    gt_lines)
            total_pred_lines = sum(len(pred_lines[file_name]) for file_name in \
                    pred_lines)
            detection_accuracy /= total_gt_lines
            recognition_accuracy /= total_pred_lines
            f_measure = 2 * detection_accuracy * recognition_accuracy / \
                    (detection_accuracy + recognition_accuracy) if \
                    detection_accuracy and recognition_accuracy else 0

            print("--- Global Results ---")
            print("DA: " + str(detection_accuracy) + 
                    ", RA: " + str(recognition_accuracy) + ", F: " + \
                            str(f_measure) + "\n")

            if results_range_output:
                results_range_file.write(str(matchscore_threshold) + "," + \
                        str(detection_accuracy) + "," + \
                        str(recognition_accuracy) + "," + str(f_measure) + "\n")
        else:
            print("No files evaluated. Check that the ground truth " + \
                    "directory exists and contains valid files.")

        if results_output:
            results_file.close()

        # End iteration if range was not specified
        if not iterate_thresh:
            break

    if results_range_output:
        results_range_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
