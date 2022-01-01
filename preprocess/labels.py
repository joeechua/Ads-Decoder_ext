import os
import json
import pickle
from sklearn.preprocessing import LabelEncoder
import argparse

# Add arguments to parser
parser = argparse.ArgumentParser(
    description="Preprocess on labels in symbols annotation"
)
parser.add_argument(
    "--label_encoder_path",
    dest="label_encoder_path",
    help="label encoder path",
    default=None,
    type=str,
)

UNCLEAR_CLUSTER_ID = 54


def get_symbol_cluster_name(filename="preprocess/clustered_symbol_list.json"):
    """Append all symbol cluster name into a list

    Args:
        filename (str, optional): path to the symbol cluster file.

    Returns:
        List[str]: a list of cluster name
    """
    cluster_names = []
    with open(filename, "r") as fp:
        data = json.loads(fp.read())
    for cluster in data["data"]:
        cluster_names.append(cluster["cluster_name"])
    return cluster_names


def load_symbol_cluster(filename="preprocess/clustered_symbol_list.json"):
    """Loads the symbol word mapping.

    Args:
        filename (str, optional): path to the symbol cluster file.

    Returns:
        word_to_id: a dict mapping from arbitrary word to symbol_id.
        id_to_symbol: a dict mapping from symbol_id to symbol name.
    """
    with open(filename, "r") as fp:
        data = json.loads(fp.read())

    word_to_id = {}
    id_to_symbol = {}

    for cluster in data["data"]:
        id_to_symbol[cluster["cluster_id"]] = cluster["cluster_name"]
        for symbol in cluster["symbols"]:
            word_to_id[symbol] = cluster["cluster_id"]
    return word_to_id, id_to_symbol


def load_symbols_annotation(filename="data/annotations/Symbols.json"):
    """Load symbols annotation

    Args:
        filename (str, optional): symbols annotation json file name

    Returns:
        Dict: a dictionary of symbols annotation
    """
    symbols = {}
    with open(filename, "r") as f:
        symbols = json.load(f)
    return symbols


def build_label_encoder():
    """Build a label encoder for the symbols annotation"""
    # load the symbols annotation
    symbols = load_symbols_annotation()
    # get all the labels in the symbols annotation
    labels = []
    for key in symbols:
        value = symbols[key]
        for data in value:
            labels.append(data[4])
    # remove duplicates in the lables
    labels = list(set(labels))
    # instantiate a label encoder
    label_encoder = LabelEncoder()
    # transform the labels into integer representation
    labels = label_encoder.fit_transform(labels)
    # store the label encoder in pickle file
    label_path = "miscellaneous/le.pickle"
    file = open(label_path, "wb")
    file.write(pickle.dumps(label_encoder))
    file.close()


def preprocess_labels(filename):
    """Preprocess the labels by mapping each label to the cluster name
    in the symbol cluster

    Args:
        filename (str): symbols annotation json file name
    """
    # load the symbols annotation
    symbols = load_symbols_annotation()
    # get the mapping functions
    word_to_id, id_to_word = load_symbol_cluster()
    # preprocess labels
    for key in symbols:
        value = symbols[key]
        for data in value:
            # map label to cluster id
            labels = [
                s.strip() for s in data[4].lower().split("/") if len(s.strip()) > 0
            ]
            labels_id = [word_to_id[s] for s in labels if s in word_to_id]
            most_common_cluster_id = (
                max(labels_id, key=labels_id.count)
                if len(labels_id)
                else UNCLEAR_CLUSTER_ID
            )
            if most_common_cluster_id != UNCLEAR_CLUSTER_ID:
                data[4] = id_to_word(most_common_cluster_id)
            else:
                # Append the first label
                data[4] = labels[0]
    # write to the file
    write_dict_to_json(filename, symbols)


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
    preprocess_labels(args.label_encoder_path)
    build_label_encoder()
