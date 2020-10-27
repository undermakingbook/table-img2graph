import numpy as np


class CellProcessor:
    """cell processor class
    """
    def __init__(self, img, cells):
        self.img = img
        self.cells = cells

    def process(self):
        """processing cells

        Returns:
            cells ([Cell]): list of correct cells
        """
        # detect relationd of each cell
        self._detect_relations(self.cells)
        # get start cell
        start_cell = self._get_start_cell(self.cells)
        # identify row/col start/end numbers
        self._detect_row_number(start_cell, 0, start_cell)
        self._detect_col_number(start_cell, 0)
        # detect image areas those correspond each cell
        self._split_img_cells(self.img, self.cells)
        return self.cells

    def _detect_relations(self, cells, thr=10):
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

    def _get_start_cell(self, cells):
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
    
    def _detect_row_number(self, now_cell, now_row, base_cell):
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
            # 0-th of rights has smallest y value among all of rights
            now_cell.sort_rights()
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
            # when processing lower cell, increment now_row(because it's one line
            # down)
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
        # 0-th of downs has smallest x value among all of downs
        now_cell.sort_downs()
        if len(now_cell.downs) == 0:
            return
        else:
            # if next_cells >= 2, select most left cell
            # cells are top-to-bottom
            next_cell = now_cell.downs[0]
        self._detect_row_number(next_cell, now_cell.row_col.y_ed + 1, next_cell)

    def _detect_col_number(self, now_cell, now_col):
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
            # 0-th of downs has smallest x value among all of downs
            now_cell.sort_downs()
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
        # 0-th of rights has smallest y value among all of rights
        now_cell.sort_rights()
        if len(now_cell.rights) == 0:
            return
        else:
            next_cell = now_cell.rights[0]
        self._detect_col_number(next_cell, now_cell.row_col.x_ed + 1)

    def _split_img_cells(self, img, cells):
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





































