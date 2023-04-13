from pathlib import Path
import os


def load_file_paths_from_list(file_list_path: Path, data_path: Path) -> list[str]:
    """Loads the file names of the original data set to be worked on using the file list."""
    file_list = []
    with open(file_list_path, "r") as f:
        for line in f:
            # remove linebreak which is the last character of the string
            file_path = data_path / line[:-1]
            file_list.append(str(file_path))
    return file_list


def load_file_paths_from_dir(data_path: Path) -> list[str]:
    file_list = []
    for f_name in os.listdir(data_path):
        if f_name == ".gitkeep":
            continue
        file_path = data_path / f_name
        file_list.append(str(file_path))
    return file_list
