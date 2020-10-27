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

# Install libraries
RUN apt-get install -y \
  # for tesseract
  tesseract-ocr \
  libtesseract-dev \
  tesseract-ocr-jpn \
  # for pdf2image
  poppler-utils

# using poetry to manage python packages
RUN pip install --upgrade pip && pip install poetry
# copying files to install python packages
# COPY requirements.txt ./
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install
# RUN pip install -r requirements.txt
