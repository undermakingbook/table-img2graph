import glob
from pdf2image import convert_from_path
# import os
import numpy as np
import cv2
import subprocess
from Table import Cell
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
    np.savetxt('13.csv', to_numpy(cells), delimiter=',', fmt='%s')


# 特許のPDF用に、画像として注釈が入っている場合の注釈文検出についても書く
# 画像内に確実に表が存在する場合、かつ1つしか表が存在しない場合を前提とする
def detect_table(api, pdf_img_path, k=0):
    # 直線検出のコードをそのまま流用して一番外側の輪郭を表として検出する
    # 1枚のpdfデータを画像に変換したという想定でやる
    # 後々複数枚のpdfをそのまま挿入しても動くようにする
    # PDF読み込み
    img = cv2.imread(pdf_img_path)
    # for tesseract
    api.SetImageFile(pdf_img_path)
    v_thr = get_v_thr(api, pdf_img_path)
    # PDF画像から罫線を抽出
    vc, hc = extract_ruled_line(img, v_thr=v_thr)
    # そもそも罫線がないなら終了
    if vc is None or hc is None:
        return None
    # セルを抽出
    cells = extract_cells(img, vc, hc)
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
        cells_ = filtering_cells(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], cells, v_thr)
        print(len(cells_))
        for j, cell in enumerate(cells_):
            # 表候補の四角形とセルがちゃんと被っている(ほぼはみ出していない)
            # かつ、セルの面積を2倍しても表候補の四角形面積を超えない(でかすぎるセルはみとめない)
            # みたいな場合ならOK
            if rect_.overlap_other(cell.coord)>0.95 and rect_.calc_area() > 2 * cell.coord.calc_area():
                cv2.imwrite('temp/cell{0}.png'.format(j), img[cell.coord.y_st:cell.coord.y_ed, cell.coord.x_st:cell.coord.x_ed])
                is_continue = True
        # すでに検査が終了した四角形領域について
        for done in dones:
            # もし、今検査中の四角形領域が、過去に検査済みの四角形領域の中に収まっている場合
            # 要するに過去に検査された四角形領域のセルに相当する場合
            if is_intersect(rect, done):
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
        cv2.imwrite('temp/temp_{0}_{1}.png'.format(k, i), img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
        # 検査終了リストに格納
        dones.append(rect)
    return glob.glob('temp/temp*.png')


def is_intersect(tb1, tb2):
    return max(tb1[0], tb2[0]) <= min(tb1[0]+tb1[2], tb2[0]+tb2[2]) \
        and max(tb1[1], tb2[1]) <= min(tb1[1]+tb1[3], tb2[1]+tb2[3])


def overlap_other(tb1, tb2):
        if not is_intersect(tb1, tb2):
            return 0
        x_st = max(tb1[0], tb2[0])
        x_ed = min(tb1[0]+tb1[2], tb2[0]+tb2[2])
        y_st = max(tb1[1], tb2[1])
        y_ed = min(tb1[1]+tb1[3], tb2[1]+tb2[3])
        # mini = Rect(x_st, x_ed, y_st, y_ed)
        return (x_st+x_ed)*(y_st+y_ed) / tb2[2]*tb2[3]
        # return mini.calc_area() / oth.calc_area()

        
def detect_table(api, pdf_img_path, k=0):
    img = cv2.imread(pdf_img_path)
    # for tesseract
    api.SetImageFile(pdf_img_path)
    v_thr = get_v_thr(api, pdf_img_path)
    vc, hc = extract_ruled_line(img, v_thr=v_thr)
    if vc is None or hc is None:
        return None
    cells = extract_cells(img, vc, hc)
    # a list of vertical and horizontal ruled line
    rects = vc + hc
    # a white image which has same shape of input
    img_white = np.full(img.shape, 255, np.uint8)
    for cell in cells:
        cv2.rectangle(img_white, (cell.coord.x_st, cell.coord.y_st), (cell.coord.x_ed, cell.coord.y_ed), (0, 0, 0), 3)
    cv2.imwrite('temp/hoge_.png', img_white)
    img_white = np.full(img.shape, 255, np.uint8)
    # draw all lines to white image
    for i, rect in enumerate(rects):
        cv2.drawContours(img_white, rects, i, (0, 0, 0), 2)
    # convert image to grayscale and invert to use findContours
    img_white = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('temp/hoge.png', img_white)
    # find cells
    contours, _ = cv2.findContours(
        img_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if approx.size < 4:
            continue
        # convert approximated contours to rectangle
        x, y, w, h = cv2.boundingRect(approx)
        # save a coordinate of rectangles 
        rects.append([x, y, w, h])
    # sorting by rectangle area (discending order) 
    rects = sorted(rects, key=lambda x: -x[2] * x[3])
    if len(rects) == 0:
        print('ダメ')
        return None
    dones = []
    for i, rect in enumerate(rects):
        rect_ = Cell.Rect(rect[0], rect[0]+rect[2], rect[1], rect[1]+rect[3])
        print(i, rect)
        if np.abs(rect[2] * rect[3] - img.shape[0] * img.shape[1]) < 100:
            continue
        is_continue = False
        print(rect_)
        cells_ = filtering_cells(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], cells, v_thr)
        print(len(cells_))
        for j, cell in enumerate(cells_):
            if rect_.overlap_other(cell.coord)>0.95 and rect_.calc_area() > 2 * cell.coord.calc_area():
                cv2.imwrite('temp/cell{0}.png'.format(j), img[cell.coord.y_st:cell.coord.y_ed, cell.coord.x_st:cell.coord.x_ed])
                is_continue = True
        for done in dones:
            if is_intersect(rect, done):
                is_continue = False
                break
        print(is_continue)
        if not is_continue:
            print('break!')
            break
        # この時点でrectの領域が表画像であるとわかる
        # ここでキャプションを探索する　tesseractでも使う？
        # 一般に表の上方向にキャプションがある(下にキャプションが付いてるやつはもうviolationだろ...)
        # 表の上方向に文字列を検索して、最もy方向で表に近い文字列群を選択、かつ、文字列のほとんどの範囲(7,8割？)が表のxに被っているものを選択
        api.SetImage(Image.fromarray(img[0:rect[1], :]))
        itoimg = {i: img for i, (img, _, _, _) in enumerate(api.GetTextlines())}
        itocd = {i: coord for i, (_, coord, _, _) in enumerate(api.GetTextlines())}
        done_cds = None
        for j in range(len(itoimg)):
            img_ = np.asarray(itoimg[j])
            cds = itocd[j]
            # padding Tesseract実行用
            img_pd = np.full((img_.shape[0] + 50, img_.shape[1] + 20), 255, np.uint8)
            img_pd[25:img_.shape[0]+25, 10:img_.shape[1]+10] = img_
            # OCR実行
            api.SetImage(Image.fromarray(img_pd))
            text = re.sub(r'\s', '', api.GetUTF8Text())
            ptn = re.compile(r'^表[0-9０-９]*[^0-9０-９【】（）\(\)\[\]\［\］]+')
            print(text)
            if ptn.search(text) is not None:
                # ここがTrueならこの段落から下を全部抽出
                # ただし、一番下の文字列の場合のみそれを行う
                done_cds = cds
        print(done_cds)
        if done_cds is not None:
            # スタートのx(これマイナス10くらいする)
            x = max(min(done_cds['x']-10, rect[0]), 0)
            # スタートのy(これマイナス25くらいする)
            y = max(done_cds['y']-25, 0)
            x_end = max(x+done_cds['w'], rect[0]+rect[2])
            cv2.imwrite('temp/temp_{0}_{1}.png'.format(k, i), img[y:rect[1]+rect[3], x:x_end])
            dones.append(rect)
    subprocess.run(['rm', *glob.glob('temp/cell*.png')])
    subprocess.run(['rm', *glob.glob('temp/hoge*.png')])
    return glob.glob('temp/temp*.png')


if __name__ == '__main__':
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your_gcp_jsonpath'
    # tesseract ocr api
    api = PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY, lang='jpn')
    # form of /path/to/table_images/
    # imgpaths = glob.glob(os.environ['TABLE_IMG_PATH'] + '*')
    # imgpaths = ['sample_image/13.png']
    imgpaths = ['sample_pat.pdf']
    subprocess.run('mkdir temp'.split())
    for path in imgpaths:
        if path.split('.')[-1] == 'pdf':
            imgs = convert_from_path(path)
            for i, img in enumerate(imgs):
                img.save('temp/input_{0}.png'.format(i), quality=95)
            for i in range(len(imgs)):
                print(i, 'temp/input_{0}.png'.format(i))
                print(detect_table(api, 'temp/input_{0}.png'.format(i), i))
                subprocess.run(['rm', *glob.glob('temp/cell*.png')])
                subprocess.run(['rm', *glob.glob('temp/temp*.png')])
        else:
            detect_table(api, path)
        # main_process(path, api)
        # subprocess.run(['rm', *glob.glob('temp/*.png')])