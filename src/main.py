from img2graph import Img2graph

import argparse

if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(description='Converting table image from scanned pdf data to graph of cells.')
    parser.add_argument('image_path', help='image path')
    args = parser.parse_args()

    # img2graph
    i2g = Img2graph('tesseract')
    i2g.execute(args.image_path)
