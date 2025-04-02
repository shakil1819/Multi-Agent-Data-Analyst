import base64

from io import BytesIO

import matplotlib.pyplot as plt

from PIL import Image


def matplotlib_from_base64(encoded: str, title: str = None, figsize: tuple = (8, 6)):
    """Convert a base64-encoded image to a matplotlib plot and display it.

    Parameters
    ----------
    encoded : str
        The base64-encoded image string.
    title : str, optional
        A title for the plot. Default is None.
    figsize : tuple, optional
        Figure size (width, height) for the plot. Default is (8, 6).

    Returns
    -------
    fig, ax : tuple
        The matplotlib figure and axes objects.

    """
    # Decode the base64 string to bytes
    img_data = base64.b64decode(encoded)

    # Load the bytes data into a BytesIO buffer
    buf = BytesIO(img_data)

    # Open the image using Pillow
    img = Image.open(buf)

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Display the image
    ax.imshow(img)
    ax.axis("off")  # Hide the axis

    if title:
        ax.set_title(title)

    # Show the plot
    plt.show()

    return fig, ax
