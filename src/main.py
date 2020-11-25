import argparse
from glob import glob
from tqdm.auto import tqdm

from table_img2graph import TableImg2graph

def correct_last_slash(path):
    if path[-1] == '/':
        return path
    else:
        return path + '/'

def make_path_list(arg, is_dir):
    if is_dir:
        return glob(correct_last_slash(arg) + '*')
    else:
        return [arg]

if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--api', default='tesseract')
    parser.add_argument('-c', '--convert')
    parser.add_argument('-e', '--extract')
    parser.add_argument('-d', '--is_dir', action='store_true')
    args = parser.parse_args()

    # table-img2graph
    ti2g = TableImg2graph(args.api)
    if args.convert:
        lst_path = make_path_list(args.convert, args.is_dir)
        for path in lst_path:
            ti2g.convert_table_img2graph(path)
    elif args.extract:
        lst_path = make_path_list(args.extract, args.is_dir)
        for path in tqdm(lst_path):
            ti2g.extract_table(path)
    else:
        print('Arguments Error!')

