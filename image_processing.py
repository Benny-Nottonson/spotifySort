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
    asarray,
    uint32,
    zeros
)
from PIL import Image, ImageFilter
from skimage.measure import label, regionprops
from spotify_api import public_get as client_get

@cache
def get_image_from_url(url: str) -> Image:
    """Gets an image from a URL and converts it to rgb"""
    response_data = client_get(url, timeout=5)
    pil_image: Image = Image.open(BytesIO(response_data.content), mode="r").convert("RGB")
    return pil_image


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
    image: Image = get_image_from_url(image_url)
    image = asarray(
            image.resize(
                (size, size),
                Image.LANCZOS,
            )
            .filter(ImageFilter.GaussianBlur(radius=blur))
            .convert("P", palette=Image.ADAPTIVE, colors=quantized_level),
            dtype=uint32,
        )
    size_threshold = round(0.01 * size * size)
    blob = label(image, connectivity=2) + 1
    blobs = regionprops(blob)
    ccv = zeros((quantized_level, 2), dtype=uint32)
    for b in blobs:
        size = b.area
        location = b.coords[0][0]
        ccv[image[location][0]][
            int(size <= size_threshold)
        ] += size
    ccv = tuple(tuple(x) for x in ccv)
    return ccv


@cache
def ccv_distance(ccv_one: tuple, ccv_two: tuple) -> float:
    """Calculates the distance between two CCV vectors"""
    ccv_one, ccv_two = array(ccv_one), array(ccv_two)
    return numpy_sum(
        [absolute(ccv_one[:, 0] - ccv_two[:, 0]) + absolute(ccv_one[:, 1] - ccv_two[:, 1])]
    )
