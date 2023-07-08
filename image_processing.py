"""Module for processing images and colors"""
from io import BytesIO
from functools import cache
from numpy import (
    sum as numpy_sum,
    absolute,
    amax,
    ndarray,
    array,
    bincount,
    count_nonzero,
)
from PIL import Image, ImageFilter
from skimage.measure import label
from spotify_api import public_get as client_get

@cache
def get_image_from_url(url: str, size: int, blur: int, quantized_level: int) -> Image:
    """Gets an image from a URL and converts it to rgb"""
    response_data = client_get(url, timeout=5)
    pil_image: Image = Image.open(BytesIO(response_data.content), mode="r").convert("RGB")
    return process_image(pil_image, size, blur, quantized_level)

def process_image(image: Image, size: int, blur: int, quantized_level: int) -> Image:
    """Processes an image by blurring, resizing, and quantizing it"""
    resized_image = image.resize((size, size), Image.Resampling.LANCZOS)
    blurred_image = resized_image.filter(ImageFilter.GaussianBlur(blur))
    if quantized_level > 0:
        quantized_image = blurred_image.quantize(quantized_level)
    return quantized_image


def blob_extract(mac_image: ndarray) -> tuple[int, ndarray]:
    """Extracts blobs from a quantized image"""
    blob: ndarray = label(mac_image, connectivity=2) + 1
    n_blobs: int = amax(blob)
    if n_blobs > 1:
        count: ndarray = bincount(blob.ravel(), minlength=n_blobs + 1)[2:]
        n_blobs += count_nonzero(count > 1)
    return n_blobs, blob


def ccv(image_url: str, size=32, blur=2, quantized_level=16) -> tuple:
    """Calculates the Color Coherence Vector of an image"""
    image: Image = get_image_from_url(image_url, size, blur, quantized_level)
    image_array = array(image)
    size_threshold = round(0.01 * size * size)
    n_blobs, blob = blob_extract(image_array)
    table = [
        [image_array[i][j], table[blob[i][j] - 1][1] + 1] if blob[i][j] != 0 else [0, 0]
        for i in range(blob.shape[0])
        for j in range(blob.shape[1])
        for table in [[[0, 0] for _ in range(0, n_blobs)]]
    ]
    color_coherence_vector = [(0, 0) for _ in range(quantized_level)]
    for color_index, blob_size in ((entry[0], entry[1]) for entry in table):
        color_coherence_vector[color_index] = (
            color_coherence_vector[color_index][0] + blob_size * (blob_size >= size_threshold),
            color_coherence_vector[color_index][1] + blob_size * (blob_size < size_threshold),
        )
    return tuple(color_coherence_vector)


@cache
def ccv_distance(ccv_one: tuple, ccv_two: tuple) -> float:
    """Calculates the distance between two CCV vectors"""
    ccv_one, ccv_two = array(ccv_one), array(ccv_two)
    return numpy_sum(
        [absolute(ccv_one[:, 0] - ccv_two[:, 0]) + absolute(ccv_one[:, 1] - ccv_two[:, 1])]
    )
