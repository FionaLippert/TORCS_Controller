#!/usr/bin/python3

import math
from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import torch
from torch.autograd import Variable
import neuralNet
from pca import PCA
import pandas as pd
import numpy as np
import os.path
import time
from sys import stdout

class MyDriver(Driver):

    def drive(self, carstate: State):
        # stdout.flush()

        t = time.time()
        # print("\033c")

        use_simple_driver = False
        use_pca = True

        command = Command()

        """
        Collect Input Data.

        Speed, Track Position, Angle on the track and the 19 values of the
        Track Edges
        """

        sensor_SPEED = carstate.speed_x * 3.6  # Convert from MPS to KPH
        sensor_TRACK_POSITION = carstate.distance_from_center
        sensor_ANGLE_TO_TRACK_AXIS = carstate.angle * math.pi / 180
        sensor_TRACK_EDGES = carstate.distances_from_edge
        # sensor_RPM = carstate.rpm

        """
        Process Inputs.

        Feed the sensor data into the network to produce a Accelrator, Brake
        and Steering command
        """

        sensor_data = [min(sensor_SPEED, 300)/300, sensor_TRACK_POSITION, sensor_ANGLE_TO_TRACK_AXIS]
        # sensor_data += list(sensor_TRACK_EDGES)

        #inputs = Variable(torch.FloatTensor(sensor_data))
        # print(inputs.size())


        if use_simple_driver:
            self.simpleDriver(carstate, command)
            print("SELF DRIVING")

        else:
            # if np.abs(sensor_TRACK_POSITION) > 1:
            #     # Off track, steer back into track with 0.0 accel (i.e. use momentum)
            #
            #     output = self.returnToTrack(sensor_ANGLE_TO_TRACK_AXIS, sensor_TRACK_POSITION)
            #
            #     print("### OFF ROAD ###")
            #
            # else:

            # PATH_TO_NEURAL_NET = "./trained_nn/mlp2.pkl"
            # PATH_TO_NEURAL_NET = "./trained_nn/esn.pkl"
            # if use_pca:
            PATH_TO_NEURAL_NET = "./trained_nn/mlp_pca.pkl"

            x = np.array(sensor_TRACK_EDGES) / 200
            if use_pca:
                y = PCA.convert(x.T)
                sensor_data += list(y)
                # print(sensor_data)
            else:
                sensor_data += list(x)

            # use MultiLayerPerceptron
            output = neuralNet.restore_MLP_and_predict(sensor_data, PATH_TO_NEURAL_NET)


            # use EchoStateNet


            # output = neuralNet.restore_ESN_and_predict(sensor_data, PATH_TO_NEURAL_NET,continuation=True)
            # try:
            #     output = self.esn.predict(sensor_data,continuation=True)
            #     # print('esn already loaded')
            # except:
            #     self.esn = neuralNet.restore_ESN(PATH_TO_NEURAL_NET)
            #     output = self.esn.predict(sensor_data,continuation=True)
                # print('load esn')
            #
            # print(output)


            if output[0] > 0:
                accel = min(max(output[0],0),1)
                brake = 0.0
            else:
                accel = 0.0
                brake = min(max(-output[0],0),1)

            steer = min(max(output[1],-1),1)

            # gear_change = output.data[3]

            print('Speed: %.2f, Track Position: %.2f, Angle to Track: %.2f\n'%(sensor_data[0], sensor_data[1], sensor_data[2]))
            print('Accelrator: %.2f, Brake: %.2f, Steering: %.2f'%(accel, brake, steer))
            print('Field View:')
            print(''.join('{:3f}, '.format(x) for x in sensor_data[3:]))



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


        """
        Automatic Transmission.

        Automatically change gears depending on the engine revs
        """

        if carstate.rpm > 8000 and command.accelerator > 0:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = max(1, carstate.gear - 1)
        if not command.gear:
            command.gear = carstate.gear or 1


        # print('current_lap_time: '+str(carstate.current_lap_time))
        # if carstate.distance_from_start <= carstate.distance_raced:
        #     print('distance_from_start: '+str(carstate.distance_from_start))
        # else:
        #     print('distance_from_start: 0')

        # print('total time: %.2fms'%((time.time()-t)*1000))

        """
        write data for evaluation to file 'simulation_log.txt'
        """
        with open('./simulation_log.txt', 'a') as file:
            file.write('current_lap_time: '+str(carstate.current_lap_time)+"\n")
            if carstate.distance_from_start <= carstate.distance_raced:
                file.write('distance_from_start: '+str(carstate.distance_from_start)+"\n")
            else:
                file.write('distance_from_start: 0 \n')


        return command
















    ############################################################################
    #
    #  Simple Driver Function for Training
    #
    ############################################################################

    def simpleDriver(self, carstate, command):

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


        else:
            new_acceleration = min(-acceleration, 1)
            command.brake = new_acceleration


        return new_acceleration

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        new_steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )
        command.steering = new_steering
        return new_steering

    def returnToTrack(self, angle, position):
        # returns commands that should bring car back onto the road
        # NEEDS WORK
        target_angle = np.pi / 4

        if position < -1:
            if (angle < np.pi / 2) & (angle > -target_angle):
                st = 1
            else:
                st = -1

            dif = np.abs(-target_angle - angle)
            st *= min(dif, target_angle) / target_angle

        else:
            if (angle < target_angle) & (angle > -np.pi / 2):
                st = -0.6
            else:
                st = 0.6

            dif = np.abs(target_angle - angle)
            st *= min(dif, target_angle) / target_angle

        # st = np.abs(sensor_TRACK_POSITION) - 1
        # st *= - 1.5 * np.sign(sensor_TRACK_POSITION)
        ac = 0.2
        br = 0.0

        return [ac, br, st]
