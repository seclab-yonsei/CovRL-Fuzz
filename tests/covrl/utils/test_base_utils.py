import pytest
import tempfile
from time import sleep
import multiprocessing as mp
from unittest.mock import patch
from covrl.utils.base_utils import *
from covrl.utils.map_target_error import map_target_error


@pytest.mark.parametrize(
    "dirname",
    [
        "",
        "set_test",
    ],
)
def test_setdir(dirname):
    with tempfile.TemporaryDirectory() as temp_dir:
        created_path = setdir(temp_dir, dirname, is_reset=True)

        # Verify the directory was created/reset at the specified path
        assert os.path.exists(created_path), f"Expected path {created_path} to exist"
        assert os.path.isdir(
            created_path
        ), f"Expected path {created_path} to be a directory"
        assert created_path == os.path.join(
            temp_dir, dirname
        ), f"Expected created path to be {os.path.join(temp_dir, dirname)}, but got {created_path}"


@pytest.mark.parametrize(
    "filename, content",
    [
        ("write_test.txt", b"test"),  # Binary content
        ("write_test.txt", "test"),  # String content
    ],
)
def test_write(filename, content):
    # Use a temporary directory as the base path for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        created_path = write(temp_dir, filename, content)

        # Verify the file was created at the expected path
        expected_path = os.path.abspath(os.path.join(temp_dir, filename))
        assert (
            created_path == expected_path
        ), f"Expected {expected_path}, but got {created_path}"
        assert os.path.exists(created_path), "Expected file to be created"

        # Verify the file content matches the specified content
        with open(created_path, "rb") as f:
            file_content = f.read()
        assert file_content == (
            content.encode() if isinstance(content, str) else content
        ), f"Expected file content to be {content}, but got {file_content}"


@pytest.mark.parametrize("content", [b"first read test", b"another read test"])
def test_read(content):
    # Use a temporary directory and file to test the read function
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, "read_test.txt")

        # Write the expected content to the temporary file
        with open(temp_filepath, "wb") as f:
            f.write(content)

        # Use the read function to read the content from the file
        result = read(temp_filepath)

        # Verify the content matches the expected value
        assert result == content, f"Expected content '{content}', but got '{result}'"


def test_init_worker():
    original_handler = signal.getsignal(signal.SIGINT)
    init_worker()
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    signal.signal(signal.SIGINT, original_handler)


def test_set_seeds():
    seed = 42

    # Call set_seeds and generate random numbers to verify reproducibility
    set_seeds(seed)

    # Generate some random numbers after setting the seed
    random_val_1 = random.random()
    np_random_val_1 = np.random.rand()
    torch_random_val_1 = torch.rand(1).item()  # Using .item() to get a scalar

    # Reset the seeds and generate random numbers again
    set_seeds(seed)

    # Generate random numbers after re-seeding
    random_val_2 = random.random()
    np_random_val_2 = np.random.rand()
    torch_random_val_2 = torch.rand(1).item()

    # Assert that the random values are the same after re-seeding
    assert (
        random_val_1 == random_val_2
    ), "Random values from 'random' module do not match"
    assert (
        np_random_val_1 == np_random_val_2
    ), "Random values from 'numpy' module do not match"
    assert (
        torch_random_val_1 == torch_random_val_2
    ), "Random values from 'torch' module do not match"


@pytest.mark.parametrize(
    "hex_encoded, expected",
    [(b"\x00\x01\x00\x02\x00\x04", [256, 512, 1024]), (b"\xd2\x04", [1234]), (b"", [])],
)
def test_hex_to_dec(hex_encoded, expected):
    dec_decoded = hex_to_dec(hex_encoded)
    assert dec_decoded == expected, f"Expected {expected}, but got {dec_decoded}"


@pytest.mark.parametrize(
    "token_ids, expected",
    [([256, 512, 1024], b"\x00\x01\x00\x02\x00\x04"), ([1234], b"\xd2\x04"), ([], b"")],
)
def test_dec_to_hex(token_ids, expected):
    hex_encoded = dec_to_hex(token_ids)
    assert hex_encoded == expected, f"Expected {expected}, but got {hex_encoded}"


def test_get_device_cpu():
    # Mock the scenario where CUDA is unavailable
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.cuda.device_count", return_value=0
    ):
        device = get_device()
        assert device.type == "cpu", "Expected CPU device when CUDA is unavailable"


def test_get_device_gpu():
    # Mock the scenario where CUDA is available
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=2
    ):
        device = get_device()
        assert device.type == "cuda", "Expected CUDA device when CUDA is available"


@pytest.mark.parametrize(
    "file_name, file_format, expected",
    [
        ("example.txt", ".txt", True),
        ("example.txt", ".pdf", False),
        ("", ".txt", False),
    ],
)
def test_check_format(file_name, file_format, expected):
    assert (
        check_format(file_name, file_format) == expected
    ), f"Expected True for {file_format} file"


@pytest.mark.parametrize(
    "error_text, error_type",
    [
        (b"SyntaxError: Error A occured", "syntaxError"),
        (b"ReferenceError: Error B occured", "referenceError"),
        (b"TypeError: Error C occured", "typeError"),
        (b"RangeError: Error D occured", "rangeError"),
        (b"URIError: Error E occured", "uriError"),
        (b"Error loading file...", "internalError"),
        (b"Error executing file...", "internalError"),
    ],
)
def test_is_error(error_text, error_type):
    target = map_target_error("v8")
    assert is_error(target, error_text) == error_type


@pytest.mark.parametrize(
    "file_structure, format, except_kw, expected_files",
    [
        # Single directory test: format is .py and no keywords to exclude
        (
            [
                ("test1.py", ""),
                ("test2.txt", ""),
                ("test3.py", ""),
                ("subdir/test4.py", ""),
                ("subdir/test5.py", ""),
            ],
            ".py",
            [],
            ["test1.py", "test3.py", "subdir/test4.py", "subdir/test5.py"],
        ),
        # Single directory test: format is .py, exclude keyword "test5"
        (
            [
                ("test1.py", ""),
                ("test2.txt", ""),
                ("test3.py", ""),
                ("subdir/test4.py", ""),
                ("subdir/test5.py", ""),
            ],
            ".py",
            ["test5"],
            ["test1.py", "test3.py", "subdir/test4.py"],
        ),
        # No matching format: expected an empty list as there are no .py files
        ([("test1.txt", ""), ("test2.log", "")], ".py", [], []),
    ],
)
def test_load_testsuites(file_structure, format, except_kw, expected_files):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files based on the provided structure
        for file_path, content in file_structure:
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)

        # Run the function
        result_files = load_testsuites(temp_dir, format=format, except_kw=except_kw)

        # Convert the result paths to relative paths for comparison
        result_files = [os.path.relpath(f, temp_dir) for f in result_files]
        assert sorted(result_files) == sorted(
            expected_files
        ), f"Expected {expected_files}, but got {result_files}"


def sample_func(x):
    sleep(0.1)  # Simulate some work
    return x * x


@pytest.mark.parametrize(
    "data, simulate_interrupt, expected",
    [
        (list(range(5)), False, [x * x for x in range(5)]),  # Normal execution
        (list(range(5)), True, None),  # Simulated KeyboardInterrupt
    ],
)
def test_pool_map(data, simulate_interrupt, expected):
    with mp.Pool(4) as pool:
        if simulate_interrupt:
            # Simulate KeyboardInterrupt directly in the pool_map function
            try:
                results = []
                for result in pool_map(pool, sample_func, data):
                    results.append(result)
                    raise KeyboardInterrupt  # Simulate an interrupt after first result
            except KeyboardInterrupt:
                # Check that the pool is terminated
                pool.terminate()
                pool.join()
                assert (
                    pool._state == "TERMINATE"
                ), "Pool should be terminated after KeyboardInterrupt"
        else:
            # Normal execution: Gather results and verify
            results = list(pool_map(pool, sample_func, data))
            assert (
                sorted(results) == expected
            ), f"Expected {expected}, but got {results}"


def test_extract_statement_token_id():
    token_ids = extract_special_token_id("Salesforce/codet5p-220m", [";"])
    assert token_ids == [
        31,
        274,
        1769,
        4359,
        4868,
        5621,
        7554,
        8284,
        8863,
        9747,
        10019,
        10663,
        11272,
        11430,
        12171,
        12386,
        13506,
        13636,
        13869,
        14432,
        15533,
        15549,
        19226,
        20451,
        20472,
        22938,
        23480,
        23489,
        25708,
        26028,
        26532,
        26994,
        27922,
        28005,
        30750,
        30943,
    ]
