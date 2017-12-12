import os
import sys

port = sys.argv[1]

if port != None:
    while True:
        os.system('./start.sh -p %s'%port)
