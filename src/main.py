from table_img2graph import TableImg2graph

import argparse

if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--api', default='tesseract')
    parser.add_argument('-c', '--convert')
    parser.add_argument('-e', '--extract')
    args = parser.parse_args()

    # table-img2graph
    ti2g = TableImg2graph(args.api)
    if args.convert:
        ti2g.convert_table_img2graph(args.convert)
    elif args.extract:
        ti2g.extract_table(args.extract)
    else:
        print('Arguments Error!')
