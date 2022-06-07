# cd /mnt/c/Users/purva.patidar/Desktop/MODULES/'Module 5'/Assignment_5.2/Docker_container
# docker build -t my_project .
# docker run -it <image ID>

FROM continuumio/miniconda3

WORKDIR /home/purvapatidar

COPY housing_price_prediction .

RUN conda env create --file env.yml
RUN conda run -n mle-dev pip install -e .
RUN conda run -n mle-dev python3 housing_packaged/main.py
RUN conda run -n mle-dev pytest


