# import glob
# import os
import numpy as np
import cv2
from tesserocr import PyTessBaseAPI, PSM, OEM
from extract_cell import extract_ruled_line, extract_cells, filtering_cells
from process_cell import detect_relations, detect_row_number, detect_col_number
from utils import get_v_thr, get_start_cell, split_img_cells, detect_text_on_tess, to_numpy
# from utils import detect_text_on_gcloud, create_img_for_gcloud


def main_process(path, api):
    img = cv2.imread(path)
    # for tesseract
    api.SetImageFile(path)
    v_thr = get_v_thr(api, path)
    vc, hc = extract_ruled_line(img, v_thr=v_thr)
    cells = extract_cells(img, vc, hc)
    # remove rectangles those are not cell
    cells = filtering_cells(img, cells, v_thr)
    # detect relationd of each cell
    detect_relations(cells)
    # get start cell
    start_cell = get_start_cell(cells)
    # identify row/col start/end numbers
    detect_row_number(start_cell, 0, start_cell)
    detect_col_number(start_cell, 0)
    # detect image areas those correspond each cell
    split_img_cells(img, cells)
    # if you want to use gcolud ocr, you should use below 2 line
    # img_gc = create_img_for_gcloud(img, cells)
    # detect_text_on_gcloud(img_gc, cells)
    for cell in cells:
        print(cell.idx, cell.row_col, cell.coord)
        # use tesseract to ocr
        detect_text_on_tess(api, cell)
        print(cell.text)
    print(to_numpy(cells))
    np.savetxt('sample.csv', to_numpy(cells), delimiter=',', fmt='%s')


if __name__ == '__main__':
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your_gcp_jsonpath'
    # tesseract ocr api
    api = PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY, lang='jpn')
    # form of /path/to/table_images/
    # imgpaths = glob.glob(os.environ['TABLE_IMG_PATH'] + '*')
    imgpaths = ['sample_image/sample.png']
    for path in imgpaths:
        main_process(path, api)
