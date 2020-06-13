import glob
import os
import math
import cv2
import numpy as np
from imutils.contours import sort_contours
from tesserocr import PyTessBaseAPI, PSM, OEM
from Table import Cell


def extract_ruled_line(img, v_thr=None):
    """Extract ruled lines of a table using cv2.findContours
    Args:
        img (np.ndarray): ndarray of input image read by opencv
        v_thr (int, optional): lowest value of horizontal kernel. Defaults to None.

    Returns:
        [(np.ndarray, np.ndarray)] : contours of vertical ruled line
        [(np.ndarray, np.ndarray)] : contours of horizontal ruled line
    """
    # convert image to grayscale and invert to use findContours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    height, width = gray.shape
    # make a threshold
    thresholded = cv2.threshold(blurred, 128, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # A vertical kernel to extract vertical ruled line of a table
    # if v_thr is None, set v_thr a larger one of 10px, 1/50 of height
    if v_thr is None:
        v_thr = max(10, height * 0.02)
    # lowest height of vertical line is v_thr
    vertical_kernel_height = math.ceil(v_thr)
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_kernel_height))

    # A horizontal kernel to extract horizontal ruled line of a table
    # set a larger one of 10px, 1/50 of width
    horizontal_kernel_width = math.ceil(max(10, width * 0.02))
    hori_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_kernel_width, 1))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(thresholded, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)

    vertical_contours, _ = cv2.findContours(
        vertical_lines_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_temp2 = cv2.erode(thresholded, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    horizontal_contours, _ = cv2.findContours(
        horizontal_lines_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sorting contours
    vertical_contours, vertical_bounding_boxes = sort_contours(
        vertical_contours, method="left-to-right")
    horizontal_contours, horizontal_bounding_boxes = sort_contours(
        horizontal_contours, method="top-to-bottom")

    return vertical_contours, horizontal_contours


def extract_cells(img, vcon, hcon, thr=3):
    """Extracting cells of a table using cv2.findContours
    Args:
        img (np.ndarray): ndarray of input image read by opencv
        vcon ([(np.ndarray, np.ndarray)]): contours of vertical ruled line
        hcon ([(np.ndarray, np.ndarray)]): contours of horizontal ruled line
        thr (int, optional): allowance of cell line gap. Defaults to 3.
    Returns:
        [Cell]: list of cells
    """
    # a list of vertical and horizontal ruled line
    rects = vcon + hcon
    # a white image which has same shape of input
    img_white = np.full(img.shape, 255, np.uint8)
    # draw all lines to white image
    for i, rect in enumerate(rects):
        cv2.drawContours(img_white, rects, i, (0, 0, 0), 2)
    # convert image to grayscale and invert to use findContours
    img_white = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
    # find cells
    contours, _ = cv2.findContours(
        img_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # filtering cell using heuristic rules
    # 1. length of contour is larger than 4 (the cell needs 4 sides at least)
    # 2. remove too big contour (the cell maybe smaller than a half of input image)
    # 3. horizontal and vertical length of any cell are fit in the fixed length
    # 4. any contour's arc length is smaller than arc length of input image
    contours = [c for c in contours if (len(c) >= 4 and cv2.contourArea(c) <= (img.shape[0] * img.shape[1]) // 2
                                        and (thr < c.reshape(-1, 2)[:, 1].mean() < img.shape[0] - thr)
                                        and (thr < c.reshape(-1, 2)[:, 0].mean() < img.shape[1] - thr)
                                        and cv2.arcLength(c, True) < img.shape[0] * 2 + img.shape[1] * 2)]
    cells = []
    i = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if approx.size < 4:
            continue
        # convert approximated contours to rectangle
        x, y, w, h = cv2.boundingRect(approx)
        # create cell
        cell = Cell.Cell(i, x, x + w, y, y + h)
        cells.append(cell)
    return cells


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


def detect_relations(cells, thr=10):
    """detect column and row relations of each cell
    Args:
        cells ([Cells]): list of all cells
        thr (int, optional): largest value of cell-to-cell height. Defaults to 10.
    """
    # smallest value of vertical/horizontal cell-to-cell
    v_min = min([c.coord.y_ed - c.coord.y_st for c in cells])
    h_min = min([c.coord.x_ed - c.coord.x_st for c in cells])
    for i in range(len(cells)):
        for j in range(len(cells)):
            # col relation
            # if a space of two cells is larger than 0 and smaller than threshold
            # and if y center of one cell are range of y of other cell
            if 0 <= cells[j].coord.x_st - cells[i].coord.x_ed <= thr\
               and (cells[j].coord.y_st < (cells[i].coord.y_st + cells[i].coord.y_ed) / 2 < cells[j].coord.y_ed
                    or cells[i].coord.y_st < (cells[j].coord.y_st + cells[j].coord.y_ed) / 2 < cells[i].coord.y_ed
                    or cells[j].coord.y_st < (cells[i].coord.y_st + cells[i].coord.y_st + v_min) / 2 < cells[j].coord.y_ed
                    or cells[i].coord.y_st < (cells[j].coord.y_st + cells[j].coord.y_st + v_min) / 2 < cells[i].coord.y_ed
                    or cells[j].coord.y_st < (cells[i].coord.y_ed + (cells[i].coord.y_ed - v_min)) / 2 < cells[j].coord.y_ed
                    or cells[i].coord.y_st < (cells[j].coord.y_ed + (cells[j].coord.y_ed - v_min)) / 2 < cells[i].coord.y_ed):
                # two cells are adjacent on column axis
                cells[i].add_rights([cells[j]])
                cells[j].add_lefts([cells[i]])
            # row relation
            # same rules with column relations
            elif 0 <= cells[j].coord.y_st - cells[i].coord.y_ed < thr\
                and (cells[j].coord.x_st < (cells[i].coord.x_st + cells[i].coord.x_ed) / 2 < cells[j].coord.x_ed
                     or cells[i].coord.x_st < (cells[j].coord.x_st + cells[j].coord.x_ed) / 2 < cells[i].coord.x_ed
                     or cells[j].coord.x_st < (cells[i].coord.x_st + cells[i].coord.x_ed + h_min) / 2 < cells[j].coord.x_ed
                     or cells[i].coord.x_st < (cells[j].coord.x_st + cells[j].coord.x_ed + h_min) / 2 < cells[i].coord.x_ed):
                cells[i].add_downs([cells[j]])
                cells[j].add_ups([cells[i]])


def detect_row_number(now_cell, now_row, base_cell):
    """detecting row start/end number of cell that starting 0, top-to-bottom

    Args:
        now_cell (Cell): A cell that current processing object
        now_row (int): A current row number
        base_cell (Cell): A cell that is used as a height standard to decide next right cell
    """
    def detect_row(now_cell, now_row, base_cell, thr=1.3):
        """detecting row end number using DFS

        Args:
            now_cell (Cell): A cell that current processing object
            now_row (int): A current row number
            base_cell (Cell): A cell that is used as a height standard to decide next right cell
            thr (float, optional): A multiplier to distinguish
                whether next cell larger than current cell or not. Defaults to 1.3.

        Returns:
            int: row end number of current cell
        """
        # if y_st of now_cell has not arleady initialized, set now_row
        if now_cell.row_col.y_st < 0:
            now_cell.row_col.y_st = now_row
        # now_cell does'nt have right cell, row end number is now_row
        if len(now_cell.rights) <= 0:
            now_cell.row_col.y_ed = now_row
            return now_cell.row_col.y_ed
        # if now_cell has only one right cell, row end number is max value
        # of recursion result(max of all right cell's row end numbers)
        elif len(now_cell.rights) == 1:
            max_row = max(
                now_row,
                detect_row(
                    now_cell.rights[0],
                    now_row,
                    base_cell))
        # if now_cell has two or more right cells, process with DFS from an upper cell
        # when processing lower cell, increment now_row(because it's one line down)
        else:
            max_row = now_row
            i = 0
            for right in now_cell.rights:
                # selecting next cell using base cell height
                # if next cell's height falls within the range of base cell's height,
                # move to next processing

                # if next cell's height smaller than current cell's one, move to next process
                # with new base cell(next cell is new base cell)
                if base_cell.coord.y_st <= (
                        right.coord.y_st + right.coord.y_ed) / 2 <= base_cell.coord.y_ed and (
                        base_cell.coord.y_ed - base_cell.coord.y_st) * thr >= right.coord.y_ed - right.coord.y_st:
                    max_cd = detect_row(right, now_row + i, right)
                    i += 1
                    max_row = max(max_row, max_cd)
                # if next cell's height larger than current cell's one, move to next process
                # with current base cell(no update on base cell)
                elif right.coord.y_st <= (base_cell.coord.y_st + base_cell.coord.y_ed) / 2 <= right.coord.y_ed:
                    max_cd = detect_row(right, now_row + i, base_cell)
                    i += 1
                    max_row = max(max_row, max_cd)
        now_cell.row_col.y_ed = max_row
        return now_cell.row_col.y_ed

    detect_row(now_cell, now_row, base_cell)
    if len(now_cell.downs) == 0:
        return
    else:
        # if next_cells >= 2, select most left cell
        # cells are top-to-bottom
        next_cell = now_cell.downs[0]
    detect_row_number(next_cell, now_cell.row_col.y_ed + 1, next_cell)


def detect_col_number(now_cell, now_col):
    """detecting column start/end number of cell that starting 0, left-to-right

    Args:
        now_cell (Cell): A cell that current processing object
        now_col (int): A current col number
    """
    def detect_col(now_cell, now_col):
        """detecting column end number using DFS
        Args:
            now_cell (Cell): A cell that current processing object
            now_col (int): A current col number
        Returns:
            int: column end number of current cell
        """
        if now_cell.row_col.x_st < 0:
            now_cell.row_col.x_st = now_col
        if len(now_cell.downs) == 0:
            now_cell.row_col.x_ed = now_col
            return now_cell.row_col.x_ed
        max_col = now_col
        for i, down in enumerate(now_cell.downs):
            max_cd = detect_col(down, now_col + i)
            max_col = max(max_col, max_cd)
        now_cell.row_col.x_ed = max_col
        return now_cell.row_col.x_ed

    detect_col(now_cell, now_col)
    if len(now_cell.rights) == 0:
        return
    else:
        next_cell = now_cell.rights[0]
    detect_col_number(next_cell, now_cell.row_col.x_ed + 1)


def get_start_cell(cells):
    """getting start cell of row/col number searching

    Args:
        cells ([Cells]): list of cells

    Returns:
        Cell: start cell
    """
    # calculate and select smallest euclid distance from origin (x_st-0)^2 + (y_st-0)^2
    return cells[np.argmin(
        [c.coord.x_st ** 2 + c.coord.y_st ** 2 for c in cells])]


def filtering_cells(cells, img, thr=10, area_thr=0.85):
    """removing rectangles that do not meet the requirement for a cell of a table
    Args:
        cells ([Cell]): list of cell candidate
        img (np.ndarray): input image
        thr (int, optional): threashold value. Defaults to 10.
        area_thr (float, optional): thr of rate of input image area. Defaults to 0.85
    Returns:
        [Cell]: list of correct cells
    """    
    del_list = []
    for i in range(len(cells)):
        # list of cells inside the i-th cell
        in_list = []
        # remove cells that are too large in area
        if cells[i].coord.calc_area() >= area_thr * img.shape[0] * img.shape[1]:
            del_list.append(i)
            continue
        # remove cells that are too large in width
        if cells[i].coord.x_ed - cells[i].coord.x_st >= area_thr * img.shape[1]:
            del_list.append(i)
            continue
        # remove cells that are too large in height
        if cells[i].coord.y_ed - cells[i].coord.y_st >= area_thr * img.shape[0]:
            del_list.append(i)
            continue
        if i in del_list:
            continue
        for j in range(len(cells)):
            if i == j:
                continue
            # if j-th cell completely inside the i-th cell, add i-th cell to in_list
            if cells[i].coord.x_st <= cells[j].coord.x_st <= cells[j].coord.x_ed <= cells[i].coord.x_ed\
               and cells[i].coord.y_st <= cells[j].coord.y_st <= cells[j].coord.y_ed <= cells[i].coord.y_ed:
                in_list.append(j)
        for j in in_list:
            for k in in_list:
                # If at least one cell with an index exists, outside cell is invalid
                if np.abs(j - k) == 1:
                    c1 = cells[min(j, k)]
                    c2 = cells[max(j, k)]
                    if np.abs(c1.coord.x_st -
                              (c2.coord.x_st + c2.coord.x_ed)) <= thr:
                        del_list.append(i)
                    break
        # if outside cell is valid, inside cells are invalid
        if i not in del_list:
            del_list += in_list
    cells = [cell for i, cell in enumerate(cells) if i not in del_list]
    return cells


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
