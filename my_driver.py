from pytocl.driver import Driver
from pytocl.car import State, Command
import torch

class MyDriver(Driver):

    def drive(self, carstate: State):
        global current_index

        command = Command()

        # Recieve Input Data
        sensor_SPEED = carstate.speed_x # * 3600 / 1000  # Convert from MPS to KPH
        sensor_TRACK_POSITION = carstate.distance_from_center
        sensor_ANGLE_TO_TRACK_AXIS = carstate.angle
        sensor_TRACK_EDGES = carstate.distances_from_edge

        # Automatic Transmission:
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        # Process Input
        # print(sensor_TRACK_EDGES)


        # Apply Accelrator, Brake and Steering Commands

        command.accelerator = 1.0
        command.brake = 0.0
        command.steering = 0.0

        return command
