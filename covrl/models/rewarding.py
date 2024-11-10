import math
import pickle
import subprocess
import traceback
from collections import namedtuple

from multiprocessing import Pool

import pandas as pd
from covrl.utils.base_utils import *
from covrl.utils.map_target_error import ErrorType, map_target_error

pd.set_option("mode.chained_assignment", None)
afl_showmap_path = os.path.abspath("./AFL/afl-showmap")

# Define a namedtuple to hold the check validity results
ValidityResult = namedtuple("ValidityResult", ["filepath", "savepath", "error"])
CheckResult = namedtuple("CheckResult", ["data", "error"])


class Rewarding:
    def __init__(self, conf, save_dir="./", bitmap_size=131072):
        self.conf = conf
        self.target = map_target_error(conf.target_interpreter)
        self.save_dir = save_dir
        self.emb_path = os.path.join(save_dir, "idf_embedding.bin")
        self.bitmap_size = bitmap_size
        self.map_size_pow2 = math.sqrt(bitmap_size)
        self.vocab = [i for i in range(bitmap_size)]
        self.idf = [0] * self.bitmap_size

    def set_dir(self, save_dir):
        self.save_dir = save_dir

    def save_embedding(self):
        with open(self.emb_path, "wb") as file:
            pickle.dump(self.idf, file)

    def load_embedding(self):
        with open(self.emb_path, "rb") as file:
            self.idf = pickle.load(file)

    def update(self, dataset, is_update_idf=True):
        # Load IDF embedding if it exists, otherwise continue with default IDF
        if os.path.isfile(self.emb_path):
            self.load_embedding()

        dataset = self.fit(dataset)

        # Update IDF and set is_orig if required
        if is_update_idf:
            self.update_idf(dataset, alpha=self.conf.alpha)
            dataset["is_orig"] = True

        # Calculate and return rewards
        dataset = self.get_reward(dataset)
        return dataset

    def update_idf(self, dataset, alpha=0.6):

        df = np.zeros(self.bitmap_size, dtype=int)
        orig_data = dataset[dataset["is_orig"] == True]

        bitmap_data = np.vstack(orig_data["bitmap"].values)

        df += np.sum(bitmap_data > 0, axis=0)

        total_docs = dataset.shape[0]
        new_idf = (np.log(total_docs / (1 + df)) / self.map_size_pow2) * (1 - alpha)
        self.idf = (alpha * np.array(self.idf)) + new_idf

        # Save the updated IDF embeddings
        self.save_embedding()

    def check_validity(self, param):
        # Check the validity of a file by running it with the specified engine and capturing any errors.
        def check_validity_for_engine(engine_path, filepath):
            # Run the engine on the specified filepath and checks for runtime errors.
            filename = f"cov_{os.path.basename(filepath)}"
            savepath = os.path.join(self.save_dir, "tmp", filename)

            cmd = [
                afl_showmap_path,
                "-o",
                savepath,
                "-m",
                "none",
                "-t",
                "5000",
                "--",
                engine_path,
                filepath,
            ]
            try:
                captured = subprocess.run(cmd, capture_output=True, timeout=20)
                stderr, stdout = captured.stderr.decode(), captured.stdout.decode()

                # Check for critical errors in stderr and stdout
                if "SEGV" in stderr or "assertion" in stdout:
                    return ValidityResult(filepath, savepath, False)

                stdout_error = is_error(self.target, captured.stdout)
                stderr_error = is_error(self.target, captured.stderr)
                error = (
                    stdout_error or stderr_error
                    if stdout_error or stderr_error
                    else False
                )
                return ValidityResult(filepath, savepath, error)

            except Exception as e:
                print(traceback.format_exc())
                return ValidityResult(None, None, True)

        # Unpack
        raw_dir, engine_path, data = param

        try:
            filename = f"{data['file_id']}.js"
            filepath = write(path=raw_dir, filename=filename, content=data["data"])
            # Check the validity of the file using the engine
            save_path, cov_path, check_error = check_validity_for_engine(
                engine_path, filepath
            )
        except Exception as e:
            # Handle exceptions during file write or validity check
            print(e)
            return CheckResult(None, None)

        # Initialize bitmap array to record coverage
        bitmap_arr = [0] * self.bitmap_size
        if not save_path and not cov_path:
            return CheckResult(None, check_error)

        try:
            # Read coverage map and content if available
            content = read(save_path).decode()
            cov = read(cov_path).decode()

            for line in cov.splitlines():
                edge = int(line.split(":")[0])
                bitmap_arr[edge] += 1

            # Update date with the coverage information and file path
            data.update({"bitmap": bitmap_arr, "data": content, "filepath": save_path})
        except Exception as e:
            # Handle exceptions during coverage map processing
            print("Validity Exception:", e)
            return CheckResult(None, check_error)

        return CheckResult(data, check_error)

    def fit(self, dataset):

        unprocessed_data = dataset[~dataset["is_orig"]]

        setdir(os.path.join(self.save_dir, "tmp"))
        dir_path = setdir(os.path.join(self.save_dir, "decoded"))

        data = [
            (
                dir_path,
                self.conf.interpreter_path,
                {
                    "is_orig": row["is_orig"],
                    "file_id": row["file_id"],
                    "data": row["data"],
                    "bitmap": None,
                    "reward": 0,
                },
            )
            for _, row in unprocessed_data.iterrows()
        ]

        core_count = 1
        pool = Pool(core_count, init_worker)
        results = pool_map(pool, self.check_validity, data)

        processed_data = []
        fail = 0
        total = len(data)

        for i, result in enumerate(results):
            data, check_error = result
            results.set_description(f"| +{i - fail} | -{fail} | {i}/{total}")
            if not data:
                fail += 1
                continue

            elif check_error == ErrorType.SYNTAX_ERROR.value:
                fail += 1
                data["reward"] = -1.0
            elif isinstance(check_error, str):
                fail += 1
                data["reward"] = -0.5

            data["is_orig"] = True
            processed_data.append(data)
        pool.terminate()

        df_processed = pd.DataFrame(
            processed_data, columns=["is_orig", "file_id", "data", "bitmap", "reward"]
        )

        updated_dataset = (
            dataset.set_index("file_id")
            .combine_first(df_processed.set_index("file_id"))
            .reset_index()
        )
        return updated_dataset

    def calculate_tfidf(self, bitmaps):
        score = np.dot(bitmaps, self.idf)
        log_score = np.where(score > 0, np.log(score), 0)
        return log_score

    def get_reward(self, dataset):
        # Filter entries where the reward value is 0
        mask = dataset["reward"] == 0
        bitmaps = np.vstack(dataset.loc[mask, "bitmap"])

        # Calculate tf-idf scores for each bitmap in a vectorized manner
        rewards = self.calculate_tfidf(bitmaps)

        # Apply sigmoid function and round the rewards to 2 decimal places
        sigmoid_rewards = 1 / (1 + np.exp(-rewards))
        rounded_rewards = np.round(sigmoid_rewards, 2)

        dataset.loc[mask, "reward"] = rounded_rewards

        return dataset.drop("bitmap", axis="columns")
