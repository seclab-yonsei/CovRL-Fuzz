import argparse
import os
import socket
import sys
import psutil
from struct import pack
from torch import cuda

from covrl.models.inferencer import Inferencer
from covrl.utils.config import Config
from covrl.utils.base_utils import dec_to_hex, hex_to_dec, set_seeds, write

HOST = "127.0.0.1"
MAX_FINETUNE_CNT = 1
global PREDICTION_PATH


def mask_mutation(conf, model, prediction_path):
    predict_file = os.path.join(prediction_path, "MLM_pred")
    with open(predict_file, "rb") as reader:
        try:
            texts = reader.read()
        except Exception:
            print("The file does not exist.")
            return
    data = hex_to_dec(texts)
    ret = model.inference(data, num_samples=conf.n_samples)
    hex_encoded = dec_to_hex(ret)
    write(predict_file, content=hex_encoded)
    return len(ret) * 2


def decode_data(model, prediction_path):
    predict_file = os.path.join(prediction_path, "MLM_decoded")
    with open(predict_file, "rb") as reader:
        try:
            texts = reader.read()
        except Exception:
            print("The file does not exist.")
            return
    data = hex_to_dec(texts)
    decoded_data = model.tokenizer.decode(data, skip_special_tokens=True)
    write(predict_file, content=decoded_data)
    return len(decoded_data)


def finetune(model, prediction_path):
    record_file = os.path.join(os.path.dirname(prediction_path), "MLM_Record")
    with open(record_file, "rb") as reader:
        try:
            texts = reader.read()
        except Exception:
            print("The file does not exist.")
            return
    model_path = texts.decode()
    if not model_path: return

    print(f"The model path is {model_path}")
    write(record_file, content="")
    new_path = model.finetune(prediction_path)
    write(record_file, content=new_path)


def start_server(
    conf, model_path, port, core=-1, sample_method="contrastive", mode="finetune"
):
    global PREDICTION_PATH

    if core >= 0:
        p = psutil.Process()
        p.cpu_affinity([core])

    model = Inferencer(
        conf=conf,
        model_path=model_path,
        sample_method=sample_method,
        save_dir=os.path.dirname(PREDICTION_PATH),
    )

    # initialize server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((HOST, port))
    server_sock.listen(1)
    conn, addr = server_sock.accept()

    print("Connected by CovRL-Fuzz execution module " + str(addr))
    write(
        os.path.join(os.path.dirname(PREDICTION_PATH), "MLM_Record"),
        content=model_path,
    )
    write(os.path.join(PREDICTION_PATH, "MLM_decoded"), content="")
    write(os.path.join(PREDICTION_PATH, "MLM_pred"), content="")
    conn.sendall(b"complete")
    data_len, finetune_cnt = 0, 0

    print("Start CovRL-Fuzz")
    while True:
        msg = conn.recv(1024)
        if not msg:
            break
        elif msg == b"predict":
            data_len = mask_mutation(conf, model, prediction_path=PREDICTION_PATH)
            conn.sendall(pack("<H", data_len))
        elif msg == b"decode":
            data_len = decode_data(model, prediction_path=PREDICTION_PATH)
            conn.sendall(pack("<H", data_len))
        elif msg == b"finetune":
            print("Start finetune")
            if finetune_cnt % MAX_FINETUNE_CNT == 0 and mode == "finetune":
                finetune(model, prediction_path=PREDICTION_PATH)
            finetune_cnt += 1
            conn.sendall(pack("<H", 10))

    print("Finish fuzz...")
    server_sock.close()
    conn.close()


def execute(
    conf, model_path, port, predict_path, core, sample_method=None, mode="finetune"
):
    global PREDICTION_PATH
    PREDICTION_PATH = predict_path
    print("=" * 50)
    print(f"Target : {conf.target_interpreter}")
    print(f"Loading prediction path {PREDICTION_PATH}")
    print(f"cpu core number : {core}")

    print("Loaded CovRL-Fuzz for execution module ")
    print(f"top k : {conf.n_samples}")
    print(f"alpha : {conf.alpha}")
    print("=" * 50)

    set_seeds(conf.seed)

    start_server(conf, model_path, port, core, sample_method, mode)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", required=True, help="Path to the configuration file.")
    arg_parser.add_argument("--port", required=True, type=int, help="Port number for communication.")
    arg_parser.add_argument("--cpu_core", required=False, default=-1, type=int, help="CPU core to use. Defaults to -1.")
    arg_parser.add_argument(
        "--mode",
        required=False,
        type=str,
        default="finetune",
        choices=["finetune", "no_finetune"],
        help="Mode of operation. 'finetune' applies fine-tuning, while 'no_finetune' skips it. Defaults to 'finetune'."

    )
    arg_parser.add_argument("--model_path", required=False, default="Salesforce/codet5p-220m", type=str,  help="Path to the pretrained model.")
    arg_parser.add_argument(
        "--sample_method", required=False, type=str, default="contrastive", choices=["contrastive"],
        help="Sampling method for inference. Currently supports 'contrastive' only. Defaults to 'contrastive'."
    )
    arg_parser.add_argument("--predict_path", required=True, type=str, help="Path to save prediction results.")
    args = arg_parser.parse_args(sys.argv[1:])

    config_path = args.config
    conf = Config.from_json(config_path)

    execute(
        conf,
        args.model_path,
        args.port,
        args.predict_path,
        args.cpu_core,
        args.sample_method,
        args.mode,
    )
