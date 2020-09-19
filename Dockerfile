# Base image
FROM python:3.7

# Environment
ENV HOME /home
WORKDIR $HOME
# COPY .bashrc requirements.txt $HOME/

# Install Commands
RUN apt-get update && apt-get upgrade -y \
  && apt-get install -y \
    git \
    vim

# RUN apt-get install -y software-properties-common
# install add-apt-repository
# RUN add-apt-repository -r ppa:alex-p/tesseract-ocr-devel -y
# RUN apt-get update -q && apt-get upgrade -y \
#   && apt-get install -y \
#     tesseract-ocr \
#     libtesseract-dev \
#     tesseract-ocr-jpn
RUN apt-get install -y \
  tesseract-ocr \
  libtesseract-dev \
  tesseract-ocr-jpn

# using poetry to manage python packages
RUN pip install --upgrade pip && pip install poetry
# copying files to install python packages
# COPY requirements.txt ./
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install
# RUN pip install -r requirements.txt