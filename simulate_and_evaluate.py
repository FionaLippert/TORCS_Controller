#!/usr/bin/python3

import subprocess
from subprocess import Popen, PIPE
import os
import signal
from xml.etree import ElementTree as et
import xml.etree.cElementTree as etc
import shutil
from os import path
import numpy as np


tracks = ['practice_eroad.xml','practice_dirt4.xml','practice_etrack2.xml']
tracks_two_cars = ['quickrace_eroad.xml','quickrace_dirt4.xml','quickrace_etrack2.xml']



def copy_config_file_to_EA_folder(old_file_name, new_file_name,subfolder_name):

    src_dir= path.abspath(os.curdir)
    dst_dir= path.join(src_dir , subfolder_name)
    dst_file = path.join(dst_dir, old_file_name)

    # create subfolder if it doesn't already exist
    os.system("mkdir -p "+dst_dir)
    os.system("chmod 777 "+dst_dir)

    # delete all .xml files in the folder
    for file in os.scandir(dst_dir):
        if file.name.endswith(".xml"):
            os.unlink(file.path)

    src_file = path.join(src_dir, old_file_name)
    shutil.copy2(src_file,dst_dir)

    dst_file = path.join(dst_dir, old_file_name)
    new_dst_file_name = path.join(dst_dir, new_file_name)
    os.rename(dst_file, new_dst_file_name)
    os.system("chmod 777 "+new_dst_file_name)

def simulate_track(track_index):

    # overwrite existing file
    with open('./simulation_log.txt', 'w') as file:
        file.write("----------New simulation-----------\n")


    # change config file to use for race
    track_index = min(max(0,track_index),len(tracks)-1)
    copy_config_file_to_EA_folder(tracks[track_index],'practice.xml','EA_current_config_file')

    #if os.path.isfile('torcs_process.txt'):
    #    os.remove('torcs_process.txt')

    subprocess.call(["./start.sh"])

    #if os.path.isfile('torcs_process.txt'):
    #    os.remove('torcs_process.txt')

def simulate_track_two_cars(track_index):

    # overwrite existing file
    with open('./simulation_log.txt', 'w') as file:
        file.write("----------New simulation-----------\n")


    # change config file to use for race
    track_index = min(max(0,track_index),len(tracks_two_cars)-1)
    copy_config_file_to_EA_folder(tracks[track_index],'practice.xml','EA_current_config_file')

    if os.path.isfile('torcs_process.txt'):
        os.remove('torcs_process.txt')

    subprocess.call(["./start.sh","-p","3001"])

    if os.path.isfile('torcs_process.txt'):
        os.remove('torcs_process.txt')


def simulate():


    # overwrite existing file
    with open('./simulation_log.txt', 'w') as file:
        file.write("----------New simulation-----------\n")

    """
    # change config file to use for race
    track_index = min(max(0,track_index),len(tracks)-1)
    copy_config_file_to_EA_folder(tracks[track_index],'practice.xml','EA_current_config_file')
    """

    if os.path.isfile('torcs_process.txt'):
        os.remove('torcs_process.txt')

    subprocess.call(["./start.sh"])

    if os.path.isfile('torcs_process.txt'):
        os.remove('torcs_process.txt')

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
    return -1.0 # something went wrong --> penalty

def get_fitness_after_time(t):
    stop = False
    dist = -1
    overtaking = 0
    damage = -1
    dist_from_center = 0
    opponents = 0
    stopped = 0
    offroad = 0
    angle = 0
    with open('./simulation_log.txt', 'r') as file:
        for line in file:

            if line.find("current_lap_time: ") > -1:
                if float(line.partition(": ")[2]) > t:
                    stop = True

            if stop and line.find("distance_from_start: ") > -1 and dist==-1:
                dist = float(line.partition(": ")[2])

            if not stop and line.find("overtaking successful") > -1:
                overtaking += 1

            if not stop and line.find("damage: ") > -1:
                damage_car = float(line.partition(": ")[2])
                if damage_car > damage:
                    damage = damage_car

            if not stop and line.find("distance from center: ") > -1:
                dfc = abs(float(line.partition(": ")[2]))
                if dfc > 0.9:
                    dist_from_center += (dfc - 0.9)

            if not stop and line.find("distance to opponent: ") > -1:
                dopp = abs(float(line.partition(": ")[2]))
                if dopp < 2:
                    opponents += dopp

            if not stop and line.find("stopped for 3 seconds") > -1:
                stopped += 1

            if not stop and line.find("off road") > -1:
                offroad += 1

            if not stop and line.find("angle: ") > -1:
                a = abs(float(line.partition(": ")[2]))
                if a > 30:
                    angle += (a - 30)

    return dist, overtaking, damage, dist_from_center, opponents, stopped, offroad, angle


def repeat_simulations(n):
    for i in range(n):
        with open('./simulation_log.txt', 'w') as file:
            file.write("----------New simulation-----------\n")
        simulate()
        print(get_distance_after_time(10.0))
