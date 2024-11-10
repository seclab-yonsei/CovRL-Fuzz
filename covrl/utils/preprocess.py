import os.path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from covrl.utils.base_utils import (dec_to_hex, load_testsuites,
                                    read, setdir, write)

def tokenize(tokenizer_name, text):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer.encode(text)


def preprocess(tokenizer_name, loadpath, save_dir):
    testsuites = load_testsuites(loadpath)
    data_records = []

    savepath = setdir(save_dir, "testsuites")
    for testsuite_path in tqdm(testsuites):
        filename = os.path.basename(testsuite_path)
        data = read(testsuite_path).decode()

        # Collect data for DataFrame
        data_records.append({"source": filename, "data": data})

        tokens = tokenize(tokenizer_name, data)
        hex_encoded = dec_to_hex(tokens)
        write(path=savepath, filename=filename, content=hex_encoded)

    # Create and save DataFrame if a path is provided
    savepath = os.path.join(save_dir, "train_dataset.json")
    df = pd.DataFrame(data_records)
    df.to_json(savepath, index=False)
    print(f"DataFrame saved to {savepath}")
