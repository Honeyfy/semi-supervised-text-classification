""" enron2ml.py converts an ENRON file format to the ML file format needed for the interns' NLP task.
    Usage:   [enron folder] [sub-category #]
    enron folder - The folder with subfolder in which we find a list of txt files and matching cats files
                default: "..\..\data\enron_with_categories"
    sub-category - a numerical value representing the sub category that classifies the txt file as positive result
                default: "1"
    output - file is saved to "..\..\data\enron_with_categories"
            filename is: enron_ml_" + sub_category + ".csv"
"""

import os
import csv

def isSubCategoryInCatsFile(filepath, top_level, sub_category):
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            cats_line = line.split(",")
            if len(cats_line) == 3:
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

if __name__ == '__main__':
    enron_folder = r"..\..\data\enron_with_categories"
    sub_category = "1"
    filename_out = enron_folder + "\enron_ml_" + sub_category + ".csv"

    subfolders = [f.path for f in os.scandir(enron_folder) if f.is_dir()]
    outfile = open(filename_out, 'w', newline='')
    writer = csv.writer(outfile, delimiter=',')
    writer.writerow(["index", "document_id", "text", "user_id", "label_id"])

    user_id = "3"  # dummy value for output csv
    file_counter = 1
    for subfolder in subfolders:
        cats_files = [f for f in os.listdir(subfolder) if f.endswith('.cats')]
        for cats_file in cats_files:
            is_sub_category_in_category = isSubCategoryInCatsFile(subfolder + "\\" + cats_file, "1", sub_category)
            label = "1" if is_sub_category_in_category else "0"
            text = getTextFromFile(subfolder + "\\" + cats_file[:-4] + "txt")
            line_array = [str(file_counter), cats_file[:-5], text, user_id, label]
            writer.writerow(line_array)
            file_counter += 1

    outfile.close()
