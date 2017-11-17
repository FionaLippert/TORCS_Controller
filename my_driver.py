import math
from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
from torch.autograd import Variable
import neuralNet

class MyDriver(Driver):

    def drive(self, carstate: State):
        command = Command()

        """
        Collect Input Data.

        Speed, Track Position, Angle on the track and the 19 values of the
        Track Edges
        """

        sensor_SPEED = carstate.speed_x * 3600 / 1000  # Convert from MPS to KPH
        sensor_TRACK_POSITION = carstate.distance_from_center
        sensor_ANGLE_TO_TRACK_AXIS = carstate.angle * math.pi / 180
        sensor_TRACK_EDGES = carstate.distances_from_edge

        """
        Automatic Transmission.

        Automatically change gears depending on the engine revs
        """

        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = max(1, carstate.gear - 1)
        if not command.gear:
            command.gear = carstate.gear or 1

        """
        Process Inputs.

        Feed the sensor data into the network to produce a Accelrator, Brake
        and Steering command
        """

        sensor_data = [sensor_SPEED, sensor_TRACK_POSITION, sensor_ANGLE_TO_TRACK_AXIS]
        sensor_data += list(sensor_TRACK_EDGES)

        inputs = Variable(torch.FloatTensor(sensor_data))
        # print(inputs.size())
        output = neuralNet.restore_net_and_predict(inputs)

        print("\033c")
        print('Speed: %.2f, Track Position: %.2f, Angle to Track: %.2f\n'%(sensor_data[0], sensor_data[1], sensor_data[3]))
        print('Accelrator: %.2f, Brake: %.2f, Steering: %.2f\n'%(output.data[0], output.data[1], output.data[2]))
        print('Field View:')
        print(sensor_data[3:])

        accel = output.data[0]
        brake = output.data[1]
        steer = output.data[2]




        """
        Apply Accelrator, Brake and Steering Commands.
        """

        command.accelerator = accel
        command.brake = brake
        command.steering = steer

        return command
