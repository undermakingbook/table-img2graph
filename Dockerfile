FROM ubuntu:16.04

ENV PYTHON_VERSION 3.7.1
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv
RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
 && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && $PYENV_ROOT/plugins/python-build/install.sh \
 && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
 && rm -rf $PYENV_ROOT
# install add-apt-repository
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:alex-p/tesseract-ocr-devel -y
RUN apt-get update
RUN apt-get install -y tesseract-ocr libtesseract-dev tesseract-ocr-jpn

RUN pip install --upgrade pip
# using poetry to manage python packages
RUN pip install poetry
# copying files to install python packages
# COPY requirements.txt ./
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && poetry install
# RUN pip install -r requirements.txt