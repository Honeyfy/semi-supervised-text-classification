import re

def remove_headers(text):
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if len(line) == 0:
            break
    return "\n".join(lines[idx:])

def remove_meta_lines(text):
    template = re.compile("^\s{0,20}(Email:|Message-ID:|Date:|From:|To:|Sent|Sent by|"
                          + "Mime-Version:|Content-Type:"
                          + "|Content-Transfer-Encoding:|X-|cc|-{4}-+).*")
    lines = text.split("\n")
    indexes = []
    for idx, line in enumerate(lines):
        if not re.search(template, line):
            indexes.append(idx)
    new_lines = [lines[i] for i in indexes]
    return "\n".join(new_lines)


