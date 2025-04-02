import os
import webbrowser


def open_html_file_in_browser(file_path: str):
    """Opens an HTML file in the default web browser.

    Parameters
    ----------
    file_path : str
        The file path or URL of the HTML file to open.

    Returns
    -------
    None

    """
    # Check if the file exists if a local path is provided.
    if os.path.isfile(file_path):
        # Convert file path to a file URL
        file_url = "file://" + os.path.abspath(file_path)
    else:
        # If the file doesn't exist locally, assume it's a URL
        file_url = file_path

    webbrowser.open(file_url)
