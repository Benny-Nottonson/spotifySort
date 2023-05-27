"""This is a module used to process images and colors"""
from io import BytesIO
from functools import cache
from numpy import (
    sum as numpy_sum,
    abs as numpy_abs,
    ndarray,
    array,
    max as numpy_max,
    bincount,
    count_nonzero,
    argmin,
    empty,
    dot,
)
from PIL import Image
from skimage.measure import label
from cv2 import cvtColor, resize, COLOR_RGB2BGR
from spotify_api import public_get as client_get


def get_image_from_url(url: str) -> ndarray:
    """Gets an image from a URL and converts it to BGR"""
    response = client_get(url, timeout=5)
    img: Image = Image.open(BytesIO(response.content))
    img: ndarray = cvtColor(array(img), COLOR_RGB2BGR)
    img: ndarray = resize(img, (16, 16))
    return img


def rgb_image_to_mac(img: ndarray) -> ndarray:
    """Converts an RGB image to a MAC image"""
    pixels: ndarray = img.reshape(-1, 3)
    mac_indices: ndarray = empty(pixels.shape[0], dtype="uint8")
    for i, pixel in enumerate(pixels):
        mac_indices[i] = find_minimum_macbeth(tuple(pixel), lab_distance_3d)
    mac_image: ndarray = mac_indices.reshape(img.shape[:2])
    return mac_image


def blob_extract(mac_image: ndarray) -> tuple:
    """Extracts blobs from a MAC image"""
    blob: ndarray = label(mac_image, connectivity=1) + 1
    n_blobs: int = numpy_max(blob)
    if n_blobs > 1:
        count: ndarray = bincount(blob.ravel(), minlength=n_blobs + 1)[2:]
        n_blobs += count_nonzero(count > 1)
    return n_blobs, blob


@cache
def find_minimum_macbeth(p_entry: tuple[int, int, int], func: callable) -> int:
    """Finds the value of q_entries that minimizes the function func(p_entry, q_entry)"""
    macbeth_colors = array(
        [
            [115, 82, 68], [194, 150, 130],
            [98, 122, 157], [87, 108, 67],
            [133, 128, 177], [103, 189, 170],
            [214, 126, 44], [80, 91, 166],
            [193, 90, 99], [94, 60, 108],
            [157, 188, 64], [224, 163, 46],
            [56, 61, 150], [70, 148, 73],
            [175, 54, 60], [231, 199, 31],
            [187, 86, 149], [8, 133, 161],
            [243, 243, 242], [200, 200, 200],
            [160, 160, 160], [122, 122, 121],
            [85, 85, 85], [52, 52, 52],
        ]
    )
    distances = array([func(p_entry, tuple(q_entry)) for q_entry in macbeth_colors])
    return argmin(distances)


@cache
def bgr_to_lab(bgr_color: tuple) -> tuple:
    """Converts a BGR color to a CIELAB color"""
    bgr_color: ndarray = array(bgr_color, dtype=float) / 255.0
    bgr_color: ndarray = _bgr_to_xyz(bgr_color)
    lab_color: tuple = _xyz_to_lab(bgr_color)
    return lab_color


def lab_distance_3d(bgr_one: tuple, bgr_two: tuple) -> float:
    """Estimates the distance between two BGR colors in LAB space"""
    l_1, a_1, b_1 = bgr_to_lab(bgr_one)
    l_2, a_2, b_2 = bgr_to_lab(bgr_two)
    return abs(l_1 - l_2) + abs(a_1 - a_2) + abs(b_1 - b_2)


def ccv(image_url: str) -> tuple:
    """Calculates the Color Coherence Vector of an image"""
    image = get_image_from_url(image_url)
    threshold = round(0.01 * image.shape[0] * image.shape[1])
    mac_image = rgb_image_to_mac(image)
    number_of_blobs, blob = blob_extract(array(mac_image))
    table = [
        [mac_image[i][j], table[blob[i][j] - 1][1] + 1] if blob[i][j] != 0 else [0, 0]
        for i in range(blob.shape[0])
        for j in range(blob.shape[1])
        for table in [[[0, 0] for _ in range(0, number_of_blobs)]]
    ]
    color_coherence_vector = [(0, 0) for _ in range(24)]
    for color_index, size in ((entry[0], entry[1]) for entry in table):
        color_coherence_vector[color_index] = (
            color_coherence_vector[color_index][0] + size * (size >= threshold),
            color_coherence_vector[color_index][1] + size * (size < threshold)
        )
    return tuple(color_coherence_vector)


@cache
def ccv_distance(ccv_one: tuple, ccv_two: tuple) -> float:
    """Calculates the distance between two CCV vectors"""
    ccv_one, ccv_two = array(ccv_one), array(ccv_two)
    return numpy_sum(
        [3 * numpy_abs(ccv_one[:, 0] - ccv_two[:, 0]) + numpy_abs(ccv_one[:, 1] - ccv_two[:, 1])]
    )


def _bgr_to_xyz(bgr_color: ndarray) -> ndarray:
    """Converts a BGR color to a CIE XYZ color"""
    mask: ndarray = bgr_color > 0.04045
    bgr_color[mask] = ((bgr_color[mask] + 0.055) / 1.055) ** 2.4
    bgr_color[~mask] /= 12.92
    bgr_matrix: ndarray = array(
        [[0.1805, 0.3576, 0.4124], [0.0722, 0.7152, 0.2126], [0.9505, 0.1192, 0.0193]]
    )
    xyz: ndarray = dot(bgr_matrix, bgr_color)
    return xyz


def _xyz_to_lab(xyz: ndarray) -> tuple:
    """Converts a CIE XYZ color to a CIELAB color"""
    xyz_n: ndarray = array([0.95047, 1.0, 1.08883])
    xyz_r: ndarray = (xyz / xyz_n) ** (1 / 3)
    mask: ndarray = xyz_r <= 0.008856
    xyz_r[mask] = (7.787 * xyz_r[mask]) + (16 / 116)
    lab_color: tuple = (
        116 * xyz_r[1] - 16,
        500 * (xyz_r[0] - xyz_r[1]),
        200 * (xyz_r[1] - xyz_r[2]),
    )
    return lab_color
