class Rect:
    def __init__(self, x_st, x_ed, y_st, y_ed):
        self.__x_st = x_st
        self.__x_ed = x_ed
        self.__y_st = y_st
        self.__y_ed = y_ed

    @property
    def x_st(self):
        return self.__x_st

    @x_st.setter
    def x_st(self, x_st):
        self.__x_st = x_st

    @property
    def x_ed(self):
        return self.__x_ed

    @x_ed.setter
    def x_ed(self, x_ed):
        self.__x_ed = x_ed

    @property
    def y_st(self):
        return self.__y_st

    @y_st.setter
    def y_st(self, y_st):
        self.__y_st = y_st

    @property
    def y_ed(self):
        return self.__y_ed

    @y_ed.setter
    def y_ed(self, y_ed):
        self.__y_ed = y_ed

    # 面積を計算

    def calc_area(self):
        return (self.__x_ed - self.__x_st) * (self.__y_ed - self.__y_st)


class Cell:
    def __init__(self, idx, x_st, x_ed, y_st, y_ed):
        # セルのインデックス
        self.__idx = idx
        # セルの座標
        self.__coord = Rect(x_st, x_ed, y_st, y_ed)
        self.row_col = Rect(-1, -1, -1, -1)
        # 各方向への隣接セル一覧
        self.__rights = []
        self.__lefts = []
        self.__ups = []
        self.__downs = []

    @property
    def idx(self):
        return self.__idx

    @property
    def coord(self):
        return self.__coord

    @property
    def rights(self):
        return self.__rights

    @property
    def lefts(self):
        return self.__lefts

    @property
    def ups(self):
        return self.__ups

    @property
    def downs(self):
        return self.__downs

    # 配列への追加はsetterじゃなくてappend形式で行う
    def add_rights(self, rights):
        self.__rights += rights

    def add_lefts(self, lefts):
        self.__lefts += lefts

    def add_ups(self, ups):
        self.__ups += ups

    def add_downs(self, downs):
        self.__downs += downs

    # 行方向へのスパニングがあるか
    def is_row_spanning(self):
        return self.__row_col.y_ed - self.__row_col.y_st > 0

    # 列方向へのスパニングがあるか

    def is_col_spanning(self):
        return self.__row_col.x_ed - self.__row_col.x_st > 0

    # TODO: 次に実装する
    # セルが行ヘッダかどうか

    def is_row_header(self):
        return False

    # TODO: 行ヘッダと同タイミングで実装する
    # セルが列ヘッダかどうか
    def is_col_header(self):
        return False
