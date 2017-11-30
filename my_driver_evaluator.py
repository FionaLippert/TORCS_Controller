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
    RECOVER_MSG = '+-'*6 + ' RECOVERING ' + '-+'*6

    period_end_time = 0
    period_start_dist = 0
    first_evaluation = True
    fitness = 0
    off_track = False   # check if off track
    recovering = False   # when recovering let robot take over
    off_time = 0   # time came off track
    recovered_time = 0   # after 3 seconds of getting back on track start evalutaing again
    warm_up = False   # in this state let robot drive
    is_stopped = False
    stopped_time = 0   # if the car is still for 3 secs go into Recovery mode



    def drive(self, carstate: State):

        # evaluation period
        EVAL_TIME = 10

        t = time.time()
        # print("\033c")

        use_simple_driver = False
        use_pca = False

        command = Command()

        '''
        Handeling Evaluation Change:
        '''
        if t > self.period_end_time and not self.recovering:
            # END CURRENT EVALUATION PERIOD:
            if self.first_evaluation:
                # check the right files and folders are available:
                if not os.path.isdir('./pool/'):
                    print('NO POOL FOLDER FOUND')
                    return command

                if not os.path.exists('./log.csv'):
                    print('Creating log.csv...')
                    open('./log.csv', 'w').close()
            else:
                # FITNESS CALCULATION
                # currently average speed (in MPS)
                # penalties for off track and recovery
                # bonus points for negotiating corners well?
                # (i.e. if steering  > 0.1 while angle < 20deg)

                self.fitness += (carstate.distance_raced - self.period_start_dist) / EVAL_TIME

                # log the performance:
                nn_id = int(self.PATH_TO_NEURAL_NET[12:-4])
                log = open('./log.csv', 'a')
                log.write('%s, %.2f\n'%(nn_id, self.fitness))
                log.close()
                print('Fitness Score %.2f logged\n'%(self.fitness))
                pass

            # START NEXT EVALUATION PERIOD:
            # load random net:
            pool = os.listdir('./pool')
            self.PATH_TO_NEURAL_NET = './pool/' + str(np.random.choice(pool, 1)[0])
            self.esn = neuralNet.restore_ESN(self.PATH_TO_NEURAL_NET)
            # reset values:
            self.fitness = 0
            self.period_end_time = t + EVAL_TIME
            self.period_start_dist = carstate.distance_raced
            self.first_evaluation = False

            print('Evaluating %s:'%self.PATH_TO_NEURAL_NET)



        """
        Collect Input Data.

        Speed, Track Position, Angle on the track and the 19 values of the
        Track Edges
        """

        sensor_SPEED = carstate.speed_x * 3.6  # Convert from MPS to KPH
        sensor_TRACK_POSITION = carstate.distance_from_center
        sensor_ANGLE_TO_TRACK_AXIS = carstate.angle * math.pi / 180
        sensor_TRACK_EDGES = carstate.distances_from_edge

        if sensor_SPEED < 2:
            if not self.is_stopped:
                self.is_stopped = True
                self.stopped_time = t
        else:
            self.is_stopped = False

        """
        Process Inputs.

        Feed the sensor data into the network to produce a Accelrator, Brake
        and Steering command
        """

        sensor_data = [min(sensor_SPEED, 300)/300, sensor_TRACK_POSITION, sensor_ANGLE_TO_TRACK_AXIS]

        if self.is_stopped & (t > self.stopped_time + 3) & (not self.recovering):
            self.recovering = True
            self.is_stopped = False
            print(self.RECOVER_MSG)
            print('Stopped for 3 seconds...')

        if self.recovering:
            # self.simpleDriver(carstate, command)
            self.simpleDriver(sensor_data, carstate, command)

            if np.abs(sensor_TRACK_POSITION) < 1:
                # recovered!
                # self.stuck = 0
                if not self.warm_up:
                    print('Back on track...')
                    self.recovered_time = t
                    self.warm_up = True

                self.off_track = False

                # considered recovered if moving fast and straightish
                if (t > self.recovered_time + 5) & (sensor_SPEED > 40) & (np.abs(sensor_ANGLE_TO_TRACK_AXIS) < 0.5):
                    self.recovering = False
                    self.warm_up = False
                    self.period_end_time = t  # will end evaluation period
                    print('Recovered and starting new evaulation')
                    print('+-'*18)

            else:
                self.off_track = True
                self.warm_up = False

        else:
            if np.abs(sensor_TRACK_POSITION) > 1:
                # Off track, steer back into track with 0.0 accel (i.e. use momentum)

                # output = self.returnToTrack(sensor_ANGLE_TO_TRACK_AXIS, sensor_TRACK_POSITION)
                if self.off_track == False:
                    print("### OFF ROAD ###")
                    self.off_time = t

                self.off_track = True
                self.fitness -= 0.1

                if t > self.off_time + 3:
                    # haven't recovered in 3 seconds
                    # get back on road and start new evaluation
                    self.fitness -= 100   # penalty
                    self.recovering = True
                    print(self.RECOVER_MSG)


            else:
                self.off_track = False

            x = np.array(sensor_TRACK_EDGES) / 200
            if use_pca:
                y = PCA.convert(x.T)
                sensor_data += list(y)
                # print(sensor_data)
            else:
                sensor_data += list(x)

            # use MultiLayerPerceptron
            # output = neuralNet.restore_MLP_and_predict(sensor_data, self.PATH_TO_NEURAL_NET)


            # use EchoStateNet


            # output = neuralNet.restore_ESN_and_predict(sensor_data, self.PATH_TO_NEURAL_NET,continuation=True)
            try:
                output = self.esn.predict(sensor_data,continuation=True)
                # print('esn already loaded')
            except:
                pass
            #     self.esn = neuralNet.restore_ESN(self.PATH_TO_NEURAL_NET)
            #     output = self.esn.predict(sensor_data,continuation=True)


            if output[0] > 0:
                accel = min(max(output[0],0),1)
                brake = 0.0
            else:
                accel = 0.0
                brake = min(max(-output[0],0),1)

            steer = min(max(output[1],-1),1)

            # print('Speed: %.2f, Track Position: %.2f, Angle to Track: %.2f\n'%(sensor_data[0], sensor_data[1], sensor_data[2]))
            # print('Accelrator: %.2f, Brake: %.2f, Steering: %.2f'%(accel, brake, steer))
            # print('Field View:')
            # print(''.join('{:3f}, '.format(x) for x in sensor_data[3:]))



            """
            Apply Accelrator, Brake and Steering Commands from the neural net
            """

            command.accelerator = accel
            command.brake = brake
            command.steering = steer


        """
        Automatic Transmission.

        Automatically change gears depending on the engine revs
        """

        if not self.recovering:
            if carstate.rpm > 8000 and command.accelerator > 0:
                command.gear = carstate.gear + 1
            elif carstate.rpm < 2500:
                command.gear = max(1, carstate.gear - 1)
            if not command.gear:
                command.gear = carstate.gear or 1

        return command






    ############################################################################
    #
    #  Simple Driver Function for Training
    #
    ############################################################################
    stuck = 0

    def isStuck(self, angle, carstate):

        if (carstate.speed_x < 3) & (np.abs(carstate.distance_from_center) < 0.7):
        # if (np.abs(angle) > 30/180*np.pi) & (carstate.speed_x < 5):
            if (self.stuck > 100) & (angle * carstate.distance_from_center < 0.0):
                return True
            else:
                self.stuck += 1
                return False

        else:
            stuck = 0
            return False


    # def simpleDriver(self, carstate, command):
    def simpleDriver(self, sensor_data, carstate, command):

        if self.isStuck(sensor_data[2], carstate):
            command.gear = -1
            command.steering = np.sign(sensor_data[1]) * 0.6
            command.brake = 0
            command.accelerator = 0.5
            # command.steer =
            # steering = self.steer(carstate, 0.0, command)

        else:
            if self.off_track:
                self.returnToTrack(sensor_data[2], sensor_data[1], command)

            else:
                # Drive normally
                self.steer(carstate, 0.0, command)

                ACC_LATERAL_MAX = 6400 * 5
                # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
                v_x = 80

                self.accelerate(carstate, v_x, command)
                command.gear = 1

        return command

    def accelerate(self, carstate, target_speed, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x

        if speed_error > 0:
            if carstate.speed_x > 0:
                if np.abs(carstate.distance_from_center) > 1:
                    command.accelerator = 0.3
                else:
                    command.accelerator = 0.6
            else:
                command.accelerator = 1
        else:
            command.accelerator = 0

        if carstate.rpm > 8000:
            command.gear = 2
        else:
            command.gear = 1

        # acceleration = self.acceleration_ctrl.control(
        #     speed_error,
        #     carstate.current_lap_time
        # )
        #
        # # stabilize use of gas and brake:
        # # acceleration = math.pow(acceleration, 3)
        #
        # if acceleration > 0:
        #     if abs(carstate.distance_from_center) >= 1:
        #         # off track, reduced grip:
        #         # acceleration = min(0.4, acceleration)
        #         acceleration = 0.4
        #
        #     new_acceleration = min(acceleration, 1)
        #     command.accelerator = 0.4
        #
        #
        # else:
        #     new_acceleration = min(-acceleration, 1)
        #     command.brake = new_acceleration

        return

    def steer(self, carstate, target_track_pos, command):
        new_steering = carstate.angle * math.pi / 180 - carstate.distance_from_center * 0.3

        # if in reverse steer opposite
        new_steering *= np.sign(carstate.speed_x)

        command.steering = new_steering
        return

    def returnToTrack(self, angle, position, command):
        target_angle = np.pi / 6   # 30 degrees

        if position < -1:
            if (angle < np.pi / 2) & (angle > -target_angle):
                st = 1
            else:
                st = -1

            dif = np.abs(-target_angle - angle)
            st *= min(dif, target_angle) / target_angle

        else:
            if (angle < target_angle) & (angle > -np.pi / 2):
                st = -1
            else:
                st = 1

            dif = np.abs(target_angle - angle)
            st *= min(dif, target_angle) / target_angle

        command.steering = st
        command.accelerator = 0.5
        command.brake = 0.0
        command.gear = 1

        return #command
