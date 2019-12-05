""" enron2ml.py converts an ENRON file format to the ML file format needed for the interns' NLP task.
    Usage: python enron2ml.py  [enron folder] [sub-category #] [optional: output filename]
    enron folder - The folder with subfolder in which we find a list of txt files and matching cats files
    sub-category - a numerical value representing the sub category that classifies the txt file as positive result
    output filename - quite obvious what this is
"""
import sys
import os
import csv

program_name = sys.argv[0]
arguments = sys.argv[1:]
if len(arguments) not in (2, 3):
    print("Usage: python enron2ml.py [enron folder] [sub-category #] [optional: output filename]")
    exit(-1)

enron_folder = arguments[0]
sub_category = arguments[1]

if len(arguments) == 2:
    filename_out = "enron_ml_"+sub_category+".csv"
else: # == 3
    filename_out = arguments[2]


def isSubCategoryInCatsFile(filepath, top_level, sub_category):
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            cats_line = line.split(",")
            if len(cats_line)==3:
                top_level_category = cats_line[0]
                second_level_category = cats_line[1]
                frequency = cats_line[2]
                if top_level_category == top_level:
                    if second_level_category == sub_category:
                        return True
        return False


def getTextFromFile(filepath):
    with open(filepath) as fp:
        lines = fp.read().splitlines()
        return "\n".join(lines)


subfolders = [f.path for f in os.scandir(enron_folder) if f.is_dir() ]

outfile = open(filename_out, 'w', newline='')
writer = csv.writer(outfile, delimiter=',')
writer.writerow(["index", "document_id", "text", "user_id", "label_id"])

user_id = "3" # dummy value for output csv
file_counter = 1
for subfolder in subfolders:
    cats_files = [f for f in os.listdir(subfolder) if f.endswith('.cats')]
    for cats_file in cats_files:
        is_sub_category_in_category = isSubCategoryInCatsFile(subfolder + "\\" + cats_file, "1", sub_category)
        label = "7" if is_sub_category_in_category else "6"
        text = getTextFromFile(subfolder + "\\" + cats_file[:-4] + "txt")
        line_array = [str(file_counter), cats_file[:-5], text, user_id, label]
        writer.writerow(line_array)
        file_counter += 1

outfile.close()

