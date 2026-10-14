# Shared line format for anything the platform might capture from stdout.
from time import strftime


def log(message):
    print("[veritas %s] %s" % (strftime("%H:%M:%S"), message), flush=True)
