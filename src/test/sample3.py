
all_files = []

all_files = ['a']

all_files = ['a', 'b']


###### READ THE PATHS FROM HERE:

path_in = "/Users/michele.tasca/MyStuff/Coding_stuff/vscode-extensions/vscode-reactive-jupyter/src/test/"
path_out = "/Users/michele.tasca/MyStuff/Coding_stuff/vscode-extensions/vscode-reactive-jupyter/src/test/"
replaces = {
    "subito-data-products-dev": "subito-data-products-pro", 
    "subito-data-gdpr-dev": "subito-data-gdpr-pro",
    "subito-data-infrastructure-dev": "subito-data-infrastructure-pro",
    "subito_data_products_dev": "subito_data_products_pro", 
    "subito_data_gdpr_dev": "subito_data_gdpr_pro",
    "subito_data_infrastructure_dev": "subito_data_infrastructure_pro",
}


##### Write a function that:
# - Reads all Text files in path recursively
# Replaces all strings in replaces with the corresponding value
# Saves the file in path_out with the same name

import os
from typing import List

# def replace_strings_in_files(path: str, replaces: dict, path_out: str):
    # Get all files in path:

all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path_in) for f in filenames if f.endswith(".json")]
all_files
# Read each file, replace the strings, and save to path_out:
for file in all_files:
    with open(file, "r") as f:
        text = f.read()
    for k, v in replaces.items():
        text = text.replace(k, v)
    # iF path_out does not exist, create it:


text

# replace_strings_in_files(path_in, replaces, path_out)

