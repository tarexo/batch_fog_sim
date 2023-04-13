import numpy as np
from pathlib import Path


def load_pointcloud(
    file_name: str,
    color_dict: dict,
    color_feature: int,
    d_type: type,
    num_features: int,
    intensity_multiplier: int,
) -> np.ndarray:
    """Loads a point cloud given a file name.

    Args:
        file_name (str): Name of the file containing the point cloud.

    Returns:
        np.ndarray: The loaded point cloud.
    """
    color_name = color_dict[color_feature]

    # assume bin file
    pc = np.fromfile(file_name, dtype=d_type)
    pc = pc.reshape((-1, num_features))

    pc[:, 3] = np.round(pc[:, 3] * intensity_multiplier)

    return pc


def write_pointcloud(pc: np.array, path: Path, file_name: str, type: str = "float32"):
    """Write a point cloud to a given file name.

    Args:
        pc (np.array): The point cloud to be written.
        path (Path): Path were the file should be written to.
        file_name (str): Name of the file containing the point cloud.
        type (str, optional): Type of the point cloud data. Defaults to "float32".
    """
    pc.astype(type).tofile(str(path / file_name))
