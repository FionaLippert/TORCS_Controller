#!/usr/bin/python3

import subprocess
from subprocess import Popen, PIPE
import os
import signal
from xml.etree import ElementTree as et
import xml.etree.cElementTree as etc


# ~/Documents/MScComputationalScience/CI/torcs-server/torcs-client/practice.xml
#torcs_command = ["torcs","-r",os.path.abspath("practice.xml")]
#torcs_process = subprocess.Popen(torcs_command)

#client_command = [os.path.abspath("start.sh")]
#client_process = subprocess.Popen(client_command,stdout=subprocess.PIPE,shell=True)
#client_command = [os.path.abspath("start.sh")]
#client_process = subprocess.Popen(client_command,stdout=subprocess.PIPE,shell=True)

tracks = ['e-road','forza','aalborg']

#def simulate(track_index):
def simulate():
    #path = os.path.abspath("practice.xml")
    #torcs_command = ["./simulate.sh"]
    #torcs_command = ["./start.sh"]
    #torcs_process = subprocess.Popen(torcs_command,stdout=PIPE)

    # overwrite existing file
    with open('./simulation_log.txt', 'w') as file:
        file.write("----------New simulation-----------\n")

    """
    # change track in config file
    tree = etc.parse('./practice.xml')
    root = tree.getroot()
    track_index = min(max(0,track_index),len(tracks)-1)
    for section in root.findall('section'):
        if section.get('name')=='Tracks':
            s = section.find('.//attstr')
            print(s.attrib['val'])
            s.attrib['val'] = tracks[track_index]
            print(s.attrib['val'])
    with open('./practice.xml','w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE params SYSTEM "../libs/tgf/params.dtd">')
        tree.write(f,'utf-8')
    """

    subprocess.call(["./start.sh"])
    #child_pid = torcs_process.pid


    #stop = False
    #distances = []
    #for line in iter(torcs_process.stdout.readline,''):
    #    print(line)
    """
    for line in iter(torcs_process.stdout.readline,''):
        if not stop:
            print(line)
        if line.find(b"current_lap_time: ") > -1:
            if float(line.partition(b": ")[2]) > time:
                stop = True
                #print(line)
                os.kill(child_pid,signal.SIGTERM)
        if stop and line.find(b"distance_from_start: ") > -1:
            #distances.append(float(line.partition(": ")[2]))
            return float(line.partition(b": ")[2])
    """

def get_distance_after_time(t):
    stop = False
    with open('./simulation_log.txt', 'r') as file:
        for line in file:
            if line.find("current_lap_time: ") > -1:
                if float(line.partition(": ")[2]) > t:
                    stop = True
            if stop and line.find("distance_from_start: ") > -1:
                #distances.append(float(line.partition(": ")[2]))
                return float(line.partition(": ")[2])

    #print('final distance: '+str(distances[0]))

def repeat_simulations(n):
    for i in range(n):
        with open('./simulation_log.txt', 'w') as file:
            file.write("----------New simulation-----------\n")
        simulate()
        print(get_distance_after_time(10.0))

#repeat_simulations(2)
