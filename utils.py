import numpy as np


def get_v_thr(api, imgpath):
    """getting frequent value of cell-to-cell height
    This function identify frequent value of character height using tesseract ocr
    to prevent line extracting function from misrecognition of vertical line of character
    as vertical line of cell
    Args:
        api (PyTessBaseAPI): an instance of PyTessBaseAPI
        imgpath (string): a path to image file
    Returns:
        int: threshold value of kernel to extract vertical rurled line
    """
    # get ocr result
    words = api.GetWords()
    # horizontal value of all words
    hs = [words['h'] for _, words in words]
    freq, bins = np.histogram(hs)
    # get around value of Mode of horizontal
    v_thr = np.ceil((bins[np.argmax(freq) + 1] + bins[np.argmax(freq)]) / 2)
    return v_thr


def get_start_cell(cells):
    """getting start cell of row/col number searching

    Args:
        cells ([Cells]): list of cells

    Returns:
        Cell: start cell
    """
    # calculate and select smallest euclid distance from origin (x_st-0)^2 +
    # (y_st-0)^2
    return cells[np.argmin(
        [c.coord.x_st ** 2 + c.coord.y_st ** 2 for c in cells])]