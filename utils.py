import numpy as np
import io
import cv2
import subprocess
from PIL import Image
from Table import Cell
from google.cloud import vision


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
        cells ([Cells]): list of cells\

    Returns:
        Cell: start cell
    """
    # calculate and select smallest euclid distance from origin (x_st-0)^2 +
    # (y_st-0)^2
    return cells[np.argmin(
        [c.coord.x_st ** 2 + c.coord.y_st ** 2 for c in cells])]


def split_img_cells(img, cells):
    """split image to areas those corresponds to each cell
    Args:
        img (np.ndarray): input original image
        cells ([Cell]): list of cells
    """    
    for cell in cells:
        # start coord convert to smaller integer
        # end coord convert to larger integer
        x_st = int(np.trunc(cell.coord.x_st))
        x_ed = int(np.ceil(cell.coord.x_ed))
        y_st = int(np.trunc(cell.coord.y_st))
        y_ed = int(np.ceil(cell.coord.y_ed))
        cell.img = img[y_st:y_ed, x_st:x_ed, :]


def detect_text_on_tess(api, cell):
    """detect texts those corresponds a cell using tesseract
    Args:
        api (PyTessBaseAPI): tesserocr api
        cell (Cell): cell
    """
    api.SetImage(Image.fromarray(cell.img))
    cell.text = api.GetUTF8Text()


def detect_text_on_gcloud(img_gc, cells):
    """detect texts those corresponds each cell using gcloud vision api
    Args:
        img_gc (np.ndarray): image does'nt have ruled lines(result of create_img_for_gcloud)
        cells ([Cell]): list of cells
    """    
    client = vision.ImageAnnotatorClient()
    # create temporary image file
    cv2.imwrite('temp_img.png', img_gc)
    # read image file
    with io.open('temp_img.png', 'rb') as image_file:
        content = image_file.read()
    # ocr using gcp
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # first ocr result is all of texts, so ignore 0th result
    for text in texts[1:]:
        vs = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        x_st, x_ed, y_st, y_ed = vs[0][0], vs[2][0], vs[0][1], vs[2][1]
        text_rect = Cell.Rect(x_st, x_ed, y_st, y_ed)
        for i, cell in enumerate(cells):
            # text and cell areas overlap larger than 60%
            if cell.coord.overlap_other(text_rect) > 0.6:
                cells[i].text += text.description
                break
    # remove temporary image file for gcloud
    subprocess.run('rm temp_img.png'.split())
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


def create_img_for_gcloud(img, cells):
    """get image to use as input image to gcloud ocr
    Args:
        img (np.ndarray): input original image
        cells ([Cell]): list of cells those have already set cell.img

    Returns:
        [np.ndarray]: created image that has only characters, not ruled lines
    """    
    img_new = np.full(img.shape, 255, np.uint8)
    for cell in cells:
        x1, y1 = cell.coord.x_st, cell.coord.y_st
        x1 = int(np.trunc(x1))
        y1 = int(np.trunc(y1))
        img_new[y1:y1 + cell.img.shape[0], x1:x1 + cell.img.shape[1], :] = cell.img
    return img_new
