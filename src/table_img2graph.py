import os
import subprocess
from pathlib import Path
from glob import glob

import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_path

import locale
locale.setlocale(locale.LC_ALL, 'C')
from tesserocr import PyTessBaseAPI, PSM, OEM

from Modules.cell_extractor import CellExtractor
from Modules.cell_processor import CellProcessor
from Modules.table_extractor import TableExtractor

class TableImg2graph:
    """Table image to graph converter class
    usage:
        >>> from img2graph import Img2graph
        >>> i2g = Img2Graph(<API_TYPE>)
    """
    def __init__(self, api_type):
        """constructor

        Args:
            api_type (str): Api type name('tesseract'/'google')
        """
        self.api_type = api_type
        self.api = self._set_api()

    def _set_api(self):
        """set api

        Returns:
            api (PyTessBaseAPI): tesserocr api
        """
        if self.api_type == 'tesseract':
            return PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY, lang='jpn')
        elif self.api_type == 'google':
            # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your_gcp_jsonpath'
            return None
        else:
            return None
    
    def convert_table_img2graph(self, path):
        """convert table image to graph

        Args:
            path (str): image path
        """
        img = cv2.imread(path)
        self.api.SetImageFile(path)

        extractor = CellExtractor(self.api)
        # for tesseract
        v_thr = extractor.get_v_thr()
        vc, hc = extractor.extract_ruled_line(img, v_thr=v_thr)
        cells = extractor.extract_cells(img, vc, hc)
        # remove rectangles those are not cell
        cells = extractor.filtering_cells(img, cells, v_thr)

        processor = CellProcessor(img, cells)
        cells = processor.process()

        # if you want to use gcolud ocr, you should use below 2 line
        # img_gc = create_img_for_gcloud(img, cells)
        # detect_text_on_gcloud(img_gc, cells)

        for cell in cells:
            # use tesseract to ocr
            self._detect_text_on_tess(self.api, cell)

        print(self._to_numpy(cells))

        # Output
        filename = os.path.basename(path).split('.')[0]
        np.savetxt(path.split(filename)[0] + filename + '.csv', self._to_numpy(cells), delimiter=',', fmt='%s', encoding='utf-8')

        print('Finish!')

    def extract_table(self, path):
        """extract table image from patent pdf file

        Args:
            path (str): pdf file path
        """
        # convert pdf  to image
        output_dir, output_page_dir = self._convert_pdf2image(path)

        # make save directories
        output_cell_dir = output_dir + 'cells/'
        subprocess.run(['mkdir', output_cell_dir])
        output_table_dir = output_dir + 'tables/'
        subprocess.run(['mkdir', output_table_dir])

        # extract table
        lst_pages = glob(output_page_dir + '*')
        for k, page in enumerate(lst_pages):
            extractor = TableExtractor(self.api, page, output_cell_dir, output_table_dir)
            extractor.detect_table(k)

        print('Finish!')

    def _create_img_for_gcloud(img, cells):
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

    def _detect_text_on_gcloud(self, img_gc, cells):
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

    def _detect_text_on_tess(self, api, cell):
        """detect texts those corresponds a cell using tesseract
        Args:
            api (PyTessBaseAPI): tesserocr api
            cell (Cell): cell
        """
        api.SetImage(Image.fromarray(cell.img))
        cell.text = api.GetUTF8Text()

    def _to_numpy(self, cells):
        """convert to numpy array
        Args:
            cells ([Cell]): list of cells
        """
        x_ed = max([c.row_col.x_ed for c in cells]) + 1
        y_ed = max([c.row_col.y_ed for c in cells]) + 1
        arr = np.full((y_ed, x_ed), '', dtype=object)
        for cell in cells:
            x1, x2 = cell.row_col.x_st, cell.row_col.x_ed
            y1, y2 = cell.row_col.y_st, cell.row_col.y_ed
            text = cell.text.replace('\n', '')
            arr[y1:y2 + 1, x1:x2 + 1] = np.full((y2 + 1 - y1, x2 + 1 - x1), text if text != '' else '-', dtype=object)
        return arr

    def _convert_pdf2image(self, path):
        """convert pdf to image

        Args:
            path (str): pdf file path

        Returns:
            output_dir (str): Directory path images are going to save there
            output_page_dir (str): Directory path page images saved
        """
        # make directory
        _path = Path(path)
        output_dir = str(_path.parent) + '/' + _path.stem + '/'
        subprocess.run(['mkdir', output_dir])

        # convert pdf to image
        page_imgs = convert_from_path(path)

        # save page images
        output_page_dir = output_dir + 'pages/'
        subprocess.run(['mkdir', output_page_dir])
        for i, img in enumerate(page_imgs):
            img.save(output_page_dir + 'page_{0}.png'.format(i), quality=95)

        return output_dir, output_page_dir
