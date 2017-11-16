from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import math
import torch
from torch.autograd import Variable
import neuralNet
import pandas as pd
import os.path


class MyDriver(Driver):

    use_simple_driver = True

    def drive(self, carstate: State):

        use_simple_driver = True

        command = Command()

        """
        Collect Input Data.

        Speed, Track Position, Angle on the track and the 19 values of the
        Track Edges
        """

        sensor_SPEED = carstate.speed_x # * 3600 / 1000  # Convert from MPS to KPH
        sensor_TRACK_POSITION = carstate.distance_from_center
        sensor_ANGLE_TO_TRACK_AXIS = carstate.angle
        sensor_TRACK_EDGES = carstate.distances_from_edge

        sensor_data = [sensor_SPEED, sensor_TRACK_POSITION, sensor_ANGLE_TO_TRACK_AXIS]
        sensor_data += list(sensor_TRACK_EDGES)

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


        if use_simple_driver:

            steering = self.steer(carstate, 0.0, command)

            ACC_LATERAL_MAX = 6400 * 5
            v_x = min(50, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
            #v_x = 80

            acceleration = self.accelerate(carstate, v_x, command)

            command_data = [0,0,0] # gas, brake, steering
            if acceleration > 0:
                command_data[0] = acceleration
            else:
                command_data[1] = acceleration
            command_data[2] = steering

            df = pd.DataFrame([command_data+sensor_data,])
            h = ['ACCELERATION','BRAKE','STEERING','SPEED','TRACK_POSITION','ANGLE_TO_TRACK_AXIS','TRACK_EDGE_0','TRACK_EDGE_1','TRACK_EDGE_2','TRACK_EDGE_3','TRACK_EDGE_4','TRACK_EDGE_5','TRACK_EDGE_6','TRACK_EDGE_7','TRACK_EDGE_8','TRACK_EDGE_9','TRACK_EDGE_10','TRACK_EDGE_11','TRACK_EDGE_12','TRACK_EDGE_13','TRACK_EDGE_14','TRACK_EDGE_15','TRACK_EDGE_16','TRACK_EDGE_17','TRACK_EDGE_18']
            file_name = './collected_data.csv'
            if os.path.isfile(file_name):
                df.to_csv(file_name,mode='a',header=False,index=False)
            else:
                df.to_csv(file_name,header=h,index=False)

        else:

            """
            Process Inputs.

            Feed the sensor data into the network to produce a Accelrator, Brake
            and Steering command
            """

            inputs = Variable(torch.FloatTensor(sensor_data))
            # print(inputs.size())
            output = neuralNet.restore_net_and_predict(inputs)

            # print(output.data[0])

            accel = output.data[0]
            brake = output.data[1]
            steer = output.data[2]




            """
            Apply Accelrator, Brake and Steering Commands.
            """

            command.accelerator = accel
            command.brake = brake
            command.steering = steer

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command


    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)

            new_acceleration = min(acceleration, 1)
            command.accelerator = new_acceleration

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        else:
            new_acceleration = min(-acceleration, 1)
            command.brake = new_acceleration

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        return new_acceleration

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        new_steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )
        command.steering = new_steering
        return new_steering
