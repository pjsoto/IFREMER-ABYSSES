import os
import numpy as np
import tensorflow as tf
import neptune.new as neptune


if __name__=='__main__':
    run = neptune.init(
        project="pjsotove/UnderWater-image-Segmentation",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMjI4NTlkMS0zNzE4LTRjYTEtYWMwMi02MzQzMTY3ZWI5NzUifQ==",
    )  # your credentials
    print(tf.__version__)
    print("Hello Datarmor")
