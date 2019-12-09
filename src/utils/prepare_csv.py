import re
import csv
import pandas as pd
import sys
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def remove_headers(text):
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if len(line) == 0:
            break
    return "\n".join(lines[idx:])

def remove_meta_lines(text):
    template = re.compile("^\s{0,20}(Email:|Message-ID:|Date:|From:|To:|Sent|Sent by|"
                          + "Mime-Version:|Content-Type:"
                          + "|Content-Transfer-Encoding:|X-|cc|-{4}-+|AM -{4}-+|_{4}_+).*")
    lines = text.split("\n")
    indexes = []
    for idx, line in enumerate(lines):
        if not re.search(template, line):
            indexes.append(idx)
    new_lines = [lines[i] for i in indexes]
    return "\n".join(new_lines)

def textDf_2_tokens(data,pattern):
    data['text'] = data['text'].str.split(pattern, expand=False)
    data = data.explode('text').reset_index(drop=True)
    return data

def remove_empty_lines(data):
    good_ind = data["text"] != ""
    return data[good_ind]

if __name__ == '__main__':
    CLEAN = False
    PARAGRAPH = True
    if CLEAN:
        enron_filename = r"..\..\data\enron_with_categories\enron_ml_1.csv"
        filename_out = r"..\..\data\enron_with_categories\enron_ml_1_clean.csv"
        outfile = open(filename_out, 'w', newline='')
        writer = csv.writer(outfile, delimiter=',')

        with open(enron_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    writer.writerow(["index", "document_id", "text", "user_id", "label_id"])
                    line_count += 1
                else:
                    text = row[2]
                    text = remove_headers(remove_meta_lines(text))
                    line_array = [row[0],row[1],text,row[3],row[4]]
                    writer.writerow(line_array)

    if PARAGRAPH:
        enron_filename = r"..\..\data\enron_with_categories\enron_ml_1_clean.csv"
        filename_out = r"..\..\data\enron_with_categories\enron_ml_1_clean_paragraphed.csv"

        data = pd.read_csv(enron_filename)
        data_p = textDf_2_tokens(data,"\n\n")
        data_p_no_Empty_lines = remove_empty_lines(data_p)
        data2 = data_p_no_Empty_lines.rename(columns={"label_id": "doc_label_id"}, errors="raise")
        data2.to_csv(filename_out,index=False)
