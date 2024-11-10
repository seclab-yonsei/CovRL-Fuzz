import pytest
import os
import tempfile
import pandas as pd
from transformers import AutoTokenizer
from covrl.utils.preprocess import preprocess, tokenize
from covrl.utils.base_utils import load_testsuites, read, hex_to_dec


def test_tokenize():
    tokenizer_name = "Salesforce/codet5p-220m"
    test_text = '"hello covrl"'
    tokens = tokenize(tokenizer_name, test_text)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    assert tokenizer.decode(tokens, skip_special_tokens=True) == test_text


def test_preprocess():
    with tempfile.TemporaryDirectory() as save_dir:
        tokenizer_name = "Salesforce/codet5p-220m"
        load_dir = "./tests/resources/test_load"
        preprocess(tokenizer_name, load_dir, save_dir)
        testsuites = load_testsuites(load_dir)
        processed_testsuites = load_testsuites(os.path.join(save_dir, "testsuites"))

        for testsuite_path, processed_path in zip(testsuites, processed_testsuites):
            testsuite_data = read(testsuite_path).decode()
            dec_origin = tokenize(tokenizer_name, testsuite_data)

            processed_data = read(processed_path)
            dec_processed = hex_to_dec(processed_data)
            assert dec_origin == dec_processed

        df = pd.read_json(os.path.join(save_dir, "train_dataset.json"))
        assert len(df) == 2
        assert list(df.columns) == ["source", "data"]
