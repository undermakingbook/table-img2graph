import subprocess

import numpy as np
import cv2
from Table import Cell

import locale
locale.setlocale(locale.LC_ALL, 'C')
from tesserocr import PyTessBaseAPI, PSM, OEM

from Modules.cell_extractor import CellExtractor


class TableExtractor:
    """table extractor class
    """
    def __init__(self, api, img_path, output_cell_dir, output_table_dir):
        """constructor

        Args:
            api (PyTessBaseAPI): an instance of PyTessBaseAPI
            img_path (str): path of page image
            output_cell_dir (str): directory path to save cell images
            output_table_dir (str): direcotry path to save table images
        """
        self.api = api
        self.img_path = img_path
        self.output_cell_dir = output_cell_dir
        self.output_table_dir = output_table_dir

    # 特許のPDF用に、画像として注釈が入っている場合の注釈文検出についても書く
    # 画像内に確実に表が存在する場合、かつ1つしか表が存在しない場合を前提とする
    def detect_table(self, k=0):
        # 直線検出のコードをそのまま流用して一番外側の輪郭を表として検出する
        # 1枚のpdfデータを画像に変換したという想定でやる
        # 後々複数枚のpdfをそのまま挿入しても動くようにする
        # PDF読み込み
        img = cv2.imread(self.img_path)
        # for tesseract
        self.api.SetImageFile(self.img_path)

        extractor = CellExtractor(self.api)
        # for tesseract
        v_thr = extractor.get_v_thr()
        # PDF画像から罫線を抽出
        vc, hc = extractor.extract_ruled_line(img, v_thr=v_thr)
        # そもそも罫線がないなら終了
        if vc is None or hc is None:
            return None
        # セルを抽出
        cells = extractor.extract_cells(img, vc, hc)
        # a list of vertical and horizontal ruled line
        # 縦横の罫線contourが格納されたリストを結合する
        rects = vc + hc
        # a white image which has same shape of input
        # 入力PDF画像と同じサイズの白色画像を作成
        # ↓はデバッグ用
        # img_white = np.full(img.shape, 255, np.uint8)
        # for cell in cells:
        #     cv2.rectangle(img_white, (cell.coord.x_st, cell.coord.y_st), (cell.coord.x_ed, cell.coord.y_ed), (0, 0, 0), 3)
        # cv2.imwrite('temp/hoge_.png', img_white)
        # 入力と同サイズの白色画像を用意
        img_white = np.full(img.shape, 255, np.uint8)
        # draw all lines to white image
        # すべての罫線を白色画像上に黒で書き込み
        for i, rect in enumerate(rects):
            cv2.drawContours(img_white, rects, i, (0, 0, 0), 2)
        # convert image to grayscale and invert to use findContours
        # 画像のタイプをグレスケに変更(findContoursを使用するため)
        img_white = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('temp/hoge.png', img_white)
        # 表っぽい領域を書き込んだ画像に対して、セルを探索
        contours, _ = cv2.findContours(
            img_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # findContoursの結果から四角形(セルとか、表全体の四角形とか)の領域を取得する
        rects = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if approx.size < 4:
                continue
            # convert approximated contours to rectangle
            x, y, w, h = cv2.boundingRect(approx)
            # save a coordinate of rectangles
            # 各四角形の座標をrectsの中にブチこむ(開始のx,yと幅、高さ)
            rects.append([x, y, w, h])
        # sorting by rectangle area (discending order)
        # 各四角形の面積を求めて、大きい順に並べる
        rects = sorted(rects, key=lambda x: -x[2] * x[3])
        # そもそも四角形が検出できなかったら終了
        if len(rects) == 0:
            print('ダメ')
            return None
        # 検査終了した四角形領域を格納するリスト
        dones = []
        for i, rect in enumerate(rects):
            rect_ = Cell.Rect(rect[0], rect[0]+rect[2], rect[1], rect[1]+rect[3])
            print(i, rect)
            # 表画像との面積差が小さい(ほぼ1ページぶん)の四角形の場合、間違った領域検出結果のことが多い
            # とりあえず100くらいの差しかない場合は除外する
            if np.abs(rect[2] * rect[3] - img.shape[0] * img.shape[1]) < 100:
                continue
            is_continue = False
            print(rect_)
            # 四角形内のセルを取得&セルでないものを除外
            cells_ = extractor.filtering_cells(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], cells, v_thr)
            print(len(cells_))
            for j, cell in enumerate(cells_):
                # 表候補の四角形とセルがちゃんと被っている(ほぼはみ出していない)
                # かつ、セルの面積を2倍しても表候補の四角形面積を超えない(でかすぎるセルはみとめない)
                # みたいな場合ならOK
                if rect_.overlap_other(cell.coord)>0.95 and rect_.calc_area() > 2 * cell.coord.calc_area():
                    cv2.imwrite(self.output_cell_dir + 'cell{0}.png'.format(j), img[cell.coord.y_st:cell.coord.y_ed, cell.coord.x_st:cell.coord.x_ed])
                    is_continue = True
            # すでに検査が終了した四角形領域について
            for done in dones:
                # もし、今検査中の四角形領域が、過去に検査済みの四角形領域の中に収まっている場合
                # 要するに過去に検査された四角形領域のセルに相当する場合
                if self._is_intersect(rect, done):
                    # そこで検査は打ち止めにする
                    is_continue = False
                    break
            print(is_continue)
            # セル検査中に121行目の条件を突破できないor現在検査中の領域がセルだった場合は検査打ち止め
            if not is_continue:
                print('break!')
                break
            # rect[1]はy座標の開始、rect[3]は幅
            # rect[2]はx座標の開始、rect[4]は高さ
            # として、imgのスライスで抽出する
            # tempとか付けてんじゃねえよって話ですね、すみません...
            cv2.imwrite(self.output_table_dir + 'table_{0}_{1}.png'.format(k, i), img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
            # 検査終了リストに格納
            dones.append(rect)

    def _is_intersect(self, tb1, tb2):
        return max(tb1[0], tb2[0]) <= min(tb1[0]+tb1[2], tb2[0]+tb2[2]) \
            and max(tb1[1], tb2[1]) <= min(tb1[1]+tb1[3], tb2[1]+tb2[3])












