#!/usr/bin/python3

import subprocess
from subprocess import Popen, PIPE
import os
import signal


# ~/Documents/MScComputationalScience/CI/torcs-server/torcs-client/practice.xml
#torcs_command = ["torcs","-r",os.path.abspath("practice.xml")]
#torcs_process = subprocess.Popen(torcs_command)

#client_command = [os.path.abspath("start.sh")]
#client_process = subprocess.Popen(client_command,stdout=subprocess.PIPE,shell=True)
#client_command = [os.path.abspath("start.sh")]
#client_process = subprocess.Popen(client_command,stdout=subprocess.PIPE,shell=True)

path = os.path.abspath("practice.xml")
torcs_command = ["./simulate.sh"]
torcs_process = subprocess.Popen(torcs_command,stdout=PIPE,shell=True)
child_pid = torcs_process.pid
print(child_pid)

stop = False
distances = []

for line in iter(torcs_process.stdout.readline,''):
    if not stop:
        print(line)
    if line.find("current_lap_time: ") > -1:
        if float(line.partition(": ")[2]) > 30.0:
            stop = True
            #print(line)
            os.kill(child_pid,signal.SIGTERM)
    if stop and line.find("distance_from_start: ") > -1:
        distances.append(float(line.partition(": ")[2]))

#subprocess.call('torcs -r ~/Documents/MScComputationalScience/CI/torcs-server/torcs-client/practice.xml & ./start.sh',shell=True)

#torcs_out = torcs_process.communicate()[0]
#client_out = client_process.communicate()[0]
#for line in torcs_out.splitlines():
    # process the output line by line
#    print(line)

print('final distance: '+str(distances[0]))





#for line in torcs_process.splitlines():
    # process the output line by line
#    print(line)
    #if line.find("distance_from_start: ") > -1:
    #    distances.append(float(line.partition(": ")[2]))

#print('final distance: '+str(distances[-1]))
