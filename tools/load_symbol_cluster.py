import json

UNCLEAR_CLUSTER_ID = 54

def get_symbol_cluster_name(filename="tools/clustered_symbol_list.json"):
  cluster_names = []
  with open(filename, 'r') as fp:
        data = json.loads(fp.read())
  for cluster in data['data']:
        cluster_names.append(cluster["cluster_name"])
  return cluster_names

def load_symbol_cluster(filename="tools/clustered_symbol_list.json"):
    """Loads the symbol word mapping.

    Args:
      filename: path to the symbol mapping file.

    Returns:
      word_to_id: a dict mapping from arbitrary word to symbol_id.
      id_to_symbol: a dict mapping from symbol_id to symbol name.
    """
    with open(filename, 'r') as fp:
        data = json.loads(fp.read())

    word_to_id = {}
    id_to_symbol = {}

    for cluster in data['data']:
        id_to_symbol[cluster['cluster_id']] = cluster['cluster_name']
        for symbol in cluster['symbols']:
            word_to_id[symbol] = cluster['cluster_id']
    return word_to_id, id_to_symbol
