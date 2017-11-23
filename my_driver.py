#!/usr/bin/python3

import math
from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import torch
from torch.autograd import Variable
import neuralNet
import pandas as pd
import os.path
import time
from sys import stdout

class MyDriver(Driver):

    def drive(self, carstate: State):

        use_simple_driver = False

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
        # sensor_RPM = carstate.rpm

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

        #inputs = Variable(torch.FloatTensor(sensor_data))
        # print(inputs.size())


        if use_simple_driver:

            steering = self.steer(carstate, 0.0, command)

            ACC_LATERAL_MAX = 6400 * 5
            v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
            #v_x = 80

            acceleration = self.accelerate(carstate, v_x, command)

            command_data = [0,0,0] # gas, brake, steering
            if acceleration > 0:
                command_data[0] = acceleration
            else:
                command_data[1] = acceleration
            command_data[2] = steering

            df = pd.DataFrame([command_data+sensor_data,])
            #h = ['ACCELERATION','BRAKE','STEERING','SPEED','TRACK_POSITION','ANGLE_TO_TRACK_AXIS','TRACK_EDGE_0','TRACK_EDGE_1','TRACK_EDGE_2','TRACK_EDGE_3','TRACK_EDGE_4','TRACK_EDGE_5','TRACK_EDGE_6','TRACK_EDGE_7','TRACK_EDGE_8','TRACK_EDGE_9','TRACK_EDGE_10','TRACK_EDGE_11','TRACK_EDGE_12','TRACK_EDGE_13','TRACK_EDGE_14','TRACK_EDGE_15','TRACK_EDGE_16','TRACK_EDGE_17','TRACK_EDGE_18']
            file_name = './collected_data.csv'
            if os.path.isfile(file_name):
                df.to_csv(file_name,mode='a',header=False,index=False)
            else:
                #df.to_csv(file_name,header=h,index=False)
                df.to_csv(file_name,header=False,index=False)


            print('current_lap_time: '+str(carstate.current_lap_time))
            if carstate.distance_from_start <= carstate.distance_raced:
                print('distance_from_start: '+str(carstate.distance_from_start))
            else:
                print('distance_from_start: 0')
            stdout.flush()


        else:

            PATH_TO_NEURAL_NET = "./trained_nn/esn.pkl"


            # use MultiLayerPerceptron
            #output = neuralNet.restore_MLP_and_predict(sensor_data, PATH_TO_NEURAL_NET)
            # use EchoStateNet

            t = time.time()
            #output = neuralNet.restore_ESN_and_predict(sensor_data, PATH_TO_NEURAL_NET,continuation=True)
            try:
                output = self.esn.predict(sensor_data,continuation=True)
                print('esn already loaded')
            except:
                self.esn = neuralNet.restore_ESN(PATH_TO_NEURAL_NET)
                output = self.esn.predict(sensor_data,continuation=True)
                print('load esn')
            print('total time: '+str(time.time()-t))

            print(output)


            accel = min(max(output[0,0],0),1)
            brake = min(max(output[1,0],0),1)
            steer = min(max(output[2,0],-1),1)
            # gear_change = output.data[3]

            #print("\033c")
            print('Speed: %.2f, Track Position: %.2f, Angle to Track: %.2f\n'%(sensor_data[0], sensor_data[1], sensor_data[2]))
            print('Accelrator: %.2f, Brake: %.2f, Steering: %.2f\n'%(accel, brake, steer))
            print('Field View:')
            print(sensor_data[3:])



            """
            Apply Accelrator, Brake and Steering Commands.
            """

            #if carstate.distance_from_start < 5:
            #    command.accelerator = 1
            #else:
            command.accelerator = accel
            command.brake = brake
            command.steering = steer
            # command.gear = gear_change

            print('distance raced: '+str(carstate.distance_raced))

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
            #command.accelerator = new_acceleration
            command.accelerator = 0

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
