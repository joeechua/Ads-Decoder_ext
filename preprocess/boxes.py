"""
Usage: 
    python preprocess/boxes.py --symbol_annotation_filename "data/annotations/Symbols.json"
"""
import json
import os
import argparse

# Add arguments to parser
parser = argparse.ArgumentParser(description='Preprocess on boxes in symbols annotation')
parser.add_argument('--symbol_annotation_filename', dest='symbol_annotation_filename',
                    help='symbols annotation file', default=None, type=str)


def load_symbols_annotation(root="data"):
    """Load symbols annotation

    Args:
        root (str, optional): root directory. Defaults to "data".

    Returns:
        Dict: a dictionary of symbols annotation
    """
    symbols = {}
    filename = os.path.join(root, "annotations/Symbols.json")
    with open(filename, "r") as f:
        symbols = json.load(f)
    return symbols


def get_valid_symbols_annotation():
    """Filter out data in symbols annotation with invalid bounding box coordinates

    Args:
        split (str, optional): train or test. Defaults to "train".

    Returns:
        Dict: a dictionary of valid symbols annotation
    """
    symbols = load_symbols_annotation()
    valid_symbols = {}
    for key in symbols:
        value = symbols[key]
        valid_value = []
        for data in value:
            xmin = min(data[0], data[2])
            ymin = min(data[1], data[3])
            xmax = max(data[0], data[2])
            ymax = max(data[1], data[3])
            width, height = xmax - xmin, ymax - ymin
            # Make sure that the coordinates forms a box
            if (width > 0) and (height > 0) and \
                0 <= xmin <= 501 and 0 <= ymin <= 501 and \
                    0 <= xmax <= 501 and 0 <= ymax <= 501:
                valid_value.append([xmin, ymin, xmax, ymax, data[4]])
        if len(valid_value) != 0:
            valid_symbols[key] = valid_value
    return valid_symbols


def write_dict_to_json(filename, dict):
    """Write dictionary to json file

    Args:
        filename (str): file name to write to
        dict (Dict): a dictionary
    """
    with open(filename, "w") as outfile:
        json.dump(dict, outfile)


if __name__ == "__main__":
    args = parser.parse_args()
    write_dict_to_json(args.symbol_annotation_filename,
                       get_valid_symbols_annotation())
