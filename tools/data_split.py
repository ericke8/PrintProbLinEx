import os
import numpy as np
import sys
import shutil
import argparse


class SourceDirCheck(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if os.path.isdir(values):
            source = os.path.abspath(values)
            if os.path.isdir(os.path.join(source, 'images')) and os.path.isdir(os.path.join(source, 'labels')):
                setattr(namespace, self.dest, source)
                return
            else:
                print('Subdirectories for images and labels not found')

        print('Invalid source directory')
        parser.print_help()
        sys.exit(2)


parser = argparse.ArgumentParser(
    description='Randomly selects some ratio of files from the source and moves it to output. Requires the source directory to have \'images\' and \'labels\' subdirectories, with identical filenames in each.')

parser.add_argument('-s', '--source',
                    help='source directory, with subdirectories of \'images\' and \'labels\'',
                    action=SourceDirCheck,
                    required=True)

parser.add_argument('-o', '--output',
                    help='output directory',
                    action='store',
                    required=True)

parser.add_argument('-r', '--ratio',
                    help='split ratio for amount of files to move',
                    action='store',
                    type=float,
                    required=True)

parser.add_argument('--image_ext',
                    help='extension for image files (e.g. \'.png\', \'.jpg\'), default assumes image and label are the same')

parser.add_argument('--label_ext',
                    help='extension for label files (e.g. \'.png\', \'.jpg\'), default assumes image and label are then same')

args = parser.parse_args()
ratio = args.ratio
source_dir = args.source
output_dir = os.path.abspath(args.output)

image_ext = args.image_ext
label_ext = args.label_ext
ext = False

if image_ext is None and label_ext is None:
    image_ext = ''
    label_ext = ''
elif (image_ext is not None and label_ext is None) or (image_ext is None and label_ext is not None):
    print('Extensions must be provided for both images and labels')
    parser.print_help()
    sys.exit(2)
else:
    ext = True

if source_dir is None or output_dir is None:
    print('Input and output directories must be specified')
    parser.print_help()
    sys.exit(2)

output_image_dir = os.path.join(output_dir, 'images')
output_label_dir = os.path.join(output_dir, 'labels')

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    os.mkdir(output_image_dir)
    os.mkdir(output_label_dir)
else:
    if not os.path.isdir(output_image_dir):
        os.mkdir(output_image_dir)
    if not os.path.isdir(output_label_dir):
        os.mkdir(output_label_dir)


source_image_dir = os.path.join(source_dir, 'images')
source_label_dir = os.path.join(source_dir, 'labels')

image_files = [os.path.join(source_image_dir, f) for f in os.listdir(source_image_dir) if os.path.isfile(
    os.path.join(source_image_dir, f)) and f.endswith(image_ext)]

sys.stdout = open('data_split_py.log', 'w')

print('Image Files Found:')
print(image_files)

select = np.random.choice(image_files, int(
    len(image_files) * ratio), replace=False)

print('Moving Files ...')
print('Source: ' + source_dir)
print('Destination: ' + output_dir)
for file in select:
    label_file = file.replace('images', 'labels')
    if ext:
        label_file = os.path.splitext(label_file)[0] + label_ext
    
    if not os.path.isfile(label_file):
        print('Label File Not Found! Check if file names and extensions match')
        print('Aborting')
        sys.stdout = sys.__stdout__
        print('Label File Not Found! Check if file names and extensions match')
        print('Aborting')
        sys.exit(2)

    print(file)
    shutil.move(file, output_image_dir)

    print(label_file)
    shutil.move(label_file, output_label_dir)

print('Done!')