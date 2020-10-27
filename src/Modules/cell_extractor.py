import math
import cv2
import numpy as np
from imutils.contours import sort_contours
from Table import Cell


class CellExtractor:
    """cell extractor class
    """
    def __init__(self, api):
        """constructor

        Args:
            api (PyTessBaseAPI): an instance of PyTessBaseAPI
        """
        self.api = api

    def get_v_thr(self):
        """getting frequent value of cell-to-cell height
        This function identify frequent value of character height using tesseract ocr
        to prevent line extracting function from misrecognition of vertical line of character
        as vertical line of cell

        Returns:
            v_thr (numpy.float64): threshold value of kernel to extract vertical rurled line
        """
        # get ocr result
        words = self.api.GetWords()
        # horizontal value of all words
        hs = [words['h'] for _, words in words]
        freq, bins = np.histogram(hs)
        # get around value of Mode of horizontal
        v_thr = np.ceil((bins[np.argmax(freq) + 1] + bins[np.argmax(freq)]) / 2)
        return v_thr

    def extract_ruled_line(self, img, v_thr=None):
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

        if not (len(vertical_contours) > 0 and len(horizontal_contours) > 0):
            return None, None
            
        # sorting contours
        vertical_contours, vertical_bounding_boxes = sort_contours(
            vertical_contours, method="left-to-right")
        horizontal_contours, horizontal_bounding_boxes = sort_contours(
            horizontal_contours, method="top-to-bottom")

        return vertical_contours, horizontal_contours
    
    def extract_cells(self, img, vcon, hcon, thr=3):
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
            i += 1
        return cells

    def filtering_cells(self, img, cells, v_thr, thr=10, area_thr=0.85):
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
            if cells[i].coord.calc_area() >= area_thr * \
                    img.shape[0] * img.shape[1]:
                del_list.append(i)
                continue
            # remove cells that are too small in width
            if cells[i].coord.x_ed - cells[i].coord.x_st <= v_thr / 2:
                del_list.append(i)
                continue
            # remove cells that are too large in width
            if cells[i].coord.x_ed - \
                    cells[i].coord.x_st >= area_thr * img.shape[1]:
                del_list.append(i)
                continue
            # remove cells that are too small in height
            if cells[i].coord.y_ed - cells[i].coord.y_st <= v_thr / 2:
                del_list.append(i)
                continue
            # remove cells that are too large in height
            if cells[i].coord.y_ed - \
                    cells[i].coord.y_st >= area_thr * img.shape[0]:
                del_list.append(i)
                continue
            if i in del_list:
                continue
            for j in range(len(cells)):
                if i == j:
                    continue
                # if j-th cell completely inside the i-th cell, add i-th cell to
                # in_list
                if cells[i].coord.x_st <= cells[j].coord.x_st <= cells[j].coord.x_ed <= cells[i].coord.x_ed\
                and cells[i].coord.y_st <= cells[j].coord.y_st <= cells[j].coord.y_ed <= cells[i].coord.y_ed:
                    in_list.append(j)
            for j in in_list:
                for k in in_list:
                    # If at least one cell with an index exists, outside cell is
                    # invalid
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
