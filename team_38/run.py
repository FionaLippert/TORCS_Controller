#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
import os

for root, dirs, files in os.walk('team_communication/positions'):
    for f in files:
        os.unlink(os.path.join(root, f))


if __name__ == '__main__':
    main(MyDriver())
