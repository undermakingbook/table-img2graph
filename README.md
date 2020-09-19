# table-img2graph
Converting table image from scanned pdf data to graph of cells.

Once input table image, construct a graph of cells and run OCR using tesseract/google cloud vision api automatically.

Each Cell type object has coordinates on an image, row/column start/end number of a table matrix, text, image subarea, adjacents(right, left, up, down).

List of cells can be converted to numpy array, so we can convert table image to csv, too.

※ A table image must not contain any objects other than a table! Use a different program to extract the table area as preprocess

## example
![sampleのコピー](https://user-images.githubusercontent.com/44527660/84593126-5b11d480-ae85-11ea-9dde-8f768d0acb6e.png)

execute example.py↓

![sample_result](https://user-images.githubusercontent.com/44527660/84593099-1ab25680-ae85-11ea-83f4-fa1ecd7e4e58.png)

## environment construction
```
$ git clone <REPOSITORY_URL>
$ docker-compose up -d --build
$ docker exec -it <CONTAINER_ID> bash
```

## usage
```
$ python3 main.py
```