import glob
import os
import cv2
from tesserocr import PyTessBaseAPI, PSM, OEM
from extract_cell import extract_ruled_line, extract_cells, filtering_cells
from process_cell import detect_relations, detect_row_number, detect_col_number
from utils import get_v_thr, get_start_cell


# TODO: adding example code
def main_process(path, api):
    img = cv2.imread(path)
    # for tesseract
    api.SetImageFile(path)
    v_thr = get_v_thr(api, path)
    vc, hc = extract_ruled_line(img, v_thr=v_thr)
    cells = extract_cells(img, vc, hc)
    # remove rectangles those are not cell
    cells = filtering_cells(cells)
    # detect relationd of each cell
    detect_relations(cells)
    # get start cell
    start_cell = get_start_cell(cells)
    # identify row/col start/end numbers
    detect_row_number(start_cell, 0, start_cell)
    detect_col_number(start_cell, 0)


if __name__ == '__main__':
    # tesseract ocr api
    api = PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY, lang='jpn')
    # form of /path/to/table_images/
    imgpaths = glob.glob(os.environ['TABLE_IMG_PATH'] + '*')
    for path in imgpaths:
        main_process(path, api)
