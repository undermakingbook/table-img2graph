from glob import glob

from tqdm.auto import tqdm

from table_img2graph import TableImg2graph

if __name__ == '__main__':

    ti2g = TableImg2graph('tesseract')

    lst_path = glob('/home/res/C22/*/tables/*')
    for path in tqdm(lst_path):
        try:
            ti2g.convert_table_img2graph(path)
        except:
            print('|> ERROR: ', path)

    print('Finish!')