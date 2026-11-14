"""Entry point for data-capture/main.py. From vgp/:

    python ../data-capture/main.py live --model live_model:load
"""
from signrec.live import LiveSignModel


def load():
    return LiveSignModel.load()
