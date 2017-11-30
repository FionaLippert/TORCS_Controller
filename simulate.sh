#!/bin/bash

args=("$@")
torcs -r ~/Documents/MScComputationalScience/CI/torcs-server/torcs-client/practice.xml & ./start.sh

#gnome-terminal -e "torcs -r ~/Documents/MScComputationalScience/CI/torcs-server/torcs-client/practice.xml"
#gnome-terminal -e "./start.sh -p 3001"

#gnome-terminal -e "./start.sh -p 3001" && gnome-terminal -e "./start.sh -p 3002"
