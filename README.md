# table-img2graph
Converting table image from scanned pdf data to graph of cells.

Once input table image, construct a graph of cells and run OCR using tesseract/google cloud vision api automatically.

Each Cell type object has coordinates on an image, row/column start/end number of a table matrix, text, image subarea, adjacents(right, left, up, down).

List of cells can be converted to numpy array, so we can convert table image to csv, too.

※ A table image must not contain any objects other than a table! Use a different program to extract the table area as preprocess

**Example:**

![sampleのコピー](https://user-images.githubusercontent.com/44527660/84593126-5b11d480-ae85-11ea-9dde-8f768d0acb6e.png)

↓

![sample_result](https://user-images.githubusercontent.com/44527660/84593099-1ab25680-ae85-11ea-83f4-fa1ecd7e4e58.png)

## function
1. Extract table images from scanned pdf data
1. Convert table images to graph of cells

## usage

### As CLI

```
$ python3 main.py [Options]
```

| Option                             | Function                                    |
|------------------------------------|---------------------------------------------|
| `-a`/`--api`                       | Choose OCR engine. (Default: `tesseract`)   |
| `-c`/`--convert` `<IMAGE_PATH>`    | Convert table images to graph of cells.     |
| `-e`/`--extract` `<PDF_FILE_PATH>` | Extract table images from scanned pdf data. |

### As module

Import and initialize:
```
from table_img2graph import TableImg2graph
t2g = TableImg2graph('tesseract')
```

Convert table images to graph of cells:
```
t2g.convert_table_img2graph(<IMAGE_PATH>)
```

Extract table images from scanned pdf data:
```
t2g.extract_table(<PDF_FILE_PATH>)
```

## develop

Develop in docker container:
```
$ git clone <REPOSITORY_URL>
$ docker-compose up -d --build
$ docker exec -it <CONTAINER_ID> bash
```
