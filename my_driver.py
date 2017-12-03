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
import os
import subprocess

class MyDriver(Driver):
    RECOVER_MSG = '+-'*6 + ' RECOVERING ' + '-+'*6

    last_cur_lap_time = 0
    period_end_time = -1
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
    init_time = 0   # only listen to the network after 1 second to initialise the ESN

    def __init__(self, logdata=False):
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None


        """
        launch torcs practice race when ./start.sh is executed (if torcs is not already running)
        comment out for normal use of torcs
        """
        """
        if not os.path.isfile('torcs_process.txt'):
            torcs_command = ["torcs","-r",os.path.abspath("./EA_current_config_file/practice.xml")]
            #torcs_command = ["torcs","-r",os.path.abspath("practice.xml")]
            self.torcs_process = subprocess.Popen(torcs_command)

            with open('./torcs_process.txt', 'w') as file:
                file.write('running torcs process')
            second_car_command = ["./start.sh","-p","3002"]
            subprocess.Popen(second_car_command)
        #torcs_output = subprocess.check_output(torcs_command)
        """
        self.use_simple_driver = False
        self.use_pca = False
        self.use_mlp_opponents = True

        self.previous_position = 1





    def drive(self, carstate: State):

        # stdout.flush()

        t = carstate.current_lap_time
        if t < self.last_cur_lap_time:
            # made a lap, adjust timing events accordingly!
            self.period_end_time -= carstate.last_lap_time
            self.off_time -= carstate.last_lap_time
            self.recovered_time -= carstate.last_lap_time
            self.stopped_time -= carstate.last_lap_time

        self.last_cur_lap_time = t



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
        # sensor_data += list(sensor_TRACK_EDGES)

        #inputs = Variable(torch.FloatTensor(sensor_data))
        # print(inputs.size())

        if self.is_stopped & (t > self.stopped_time + 3) & (not self.recovering):
            self.recovering = True
            self.is_stopped = False
            print(self.RECOVER_MSG)
            print('Stopped for 3 seconds...')


        if self.recovering:
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

            """
            Drive using Neural Net
            """
            # if np.abs(sensor_TRACK_POSITION) > 1:
            #     # Off track, steer back into track with 0.0 accel (i.e. use momentum)
            #
            #     output = self.returnToTrack(sensor_ANGLE_TO_TRACK_AXIS, sensor_TRACK_POSITION)
            #
            #     print("### OFF ROAD ###")
            #
            # else:


            if np.abs(sensor_TRACK_POSITION) > 1:
                if self.off_track == False:
                    print("### OFF ROAD ###")
                    self.off_time = t

                self.off_track = True

                if t > self.off_time + 3:
                    # haven't recovered in 3 seconds
                    # get back on road and start new evaluation
                    self.fitness -= 100   # penalty
                    self.recovering = True
                    print(self.RECOVER_MSG)

            else:
                self.off_track = False


            # PATH_TO_NEURAL_NET = "./trained_nn/mlp2.pkl"
            PATH_TO_NEURAL_NET = "./trained_nn/esn.pkl"
            # if use_pca:
            #PATH_TO_NEURAL_NET = "./trained_nn/mlp_pca.pkl"

            PATH_TO_MLP = "./mlp_opponents.pkl"

            x = np.array(sensor_TRACK_EDGES) / 200
            if self.use_pca:
                y = PCA.convert(x.T)
                sensor_data += list(y)
                # print(sensor_data)
            else:
                sensor_data += list(x)
            """
            use MultiLayerPerceptron
            """
            #output = neuralNet.restore_MLP_and_predict(sensor_data, PATH_TO_NEURAL_NET)


            """
            use EchoStateNet
            """
            try:
                output = self.esn.predict(sensor_data,continuation=True)
                # print('esn already loaded')
            except:
                self.esn = neuralNet.restore_ESN(PATH_TO_NEURAL_NET)
                output = self.esn.predict(sensor_data,continuation=True)



            # modifiy ESN output based on the opponents data
            # only if car is not racing at the first position
            if self.use_mlp_opponents and carstate.race_position > 1:
                opponents_data = carstate.opponents


                # if closest opponent is less than 10m away, use mlp to adjust outputs
                if min(opponents_data) < 10:
                    mlp_input = [output[0],output[1]]
                    print(opponents_data)
                    for sensor in opponents_data:
                        mlp_input.append(sensor/10.0) # normalize opponents_data to [0,1]

                    #mlp = neuralNet.restore_MLP(PATH_TO_MLP)
                    mlp = neuralNet.MLP(len(mlp_input), 20, 2, np.random.rand(len(mlp_input)+1,20), np.random.rand(21,2))
                    output = mlp.predict(np.asarray(mlp_input))




            if output[0] > 0:
                accel = min(max(output[0],0),1)
                brake = 0.0
            else:
                accel = 0.0
                brake = min(max(-output[0],0),1)

            steer = min(max(output[1],-1),1)

            # gear_change = output.data[3]
            """
            print('Speed: %.2f, Track Position: %.2f, Angle to Track: %.2f\n'%(sensor_data[0], sensor_data[1], sensor_data[2]))
            print('Accelrator: %.2f, Brake: %.2f, Steering: %.2f'%(accel, brake, steer))
            print('Field View:')
            print(''.join('{:3f}, '.format(x) for x in sensor_data[3:]))
            """



            """
            # full acceleration at the start of the race
            if carstate.distance_raced<50:
                accel = 1
                brake = 0
                steer = 0
            """

            """
            Apply Accelrator, Brake and Steering Commands.
            """
            # for the first second do not listen to the network
            # this allows it to initate it's state

            if t > self.init_time:
                command.accelerator = accel
                command.brake = brake
                command.steering = steer
            else:
                self.simpleDriver(sensor_data, carstate, command)


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


        """
        write data for evaluation to file 'simulation_log.txt'
        """
        with open('./simulation_log.txt', 'a') as file:
            file.write('current_lap_time: '+str(carstate.current_lap_time)+'\n')
            if carstate.distance_from_start <= carstate.distance_raced:
                file.write('distance_from_start: '+str(carstate.distance_from_start)+'\n')
            else:
                file.write('distance_from_start: 0 \n')

            # if overtaking was successful
            if carstate.race_position > self.previous_position:
                file.write('overtaking successful \n')

            file.write('damage: '+str(carstate.damage)+'\n')
            file.write('distance from center: '+str(carstate.distance_from_center)+'\n')

            file.write('distance to opponent: '+str(min(carstate.opponents))+'\n')


        self.previous_position = carstate.race_position

        return command









    ############################################################################
    #
    #  Simple Driver Function for Recovery
    #
    ############################################################################
    stuck = 0

    def isStuck(self, angle, carstate):

        if (carstate.speed_x < 3) & (np.abs(carstate.distance_from_center) > 0.7) & (np.abs(angle) > 20/180*np.pi):
            
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
            if carstate.speed_x >= 0:
                if np.abs(carstate.distance_from_center) > 1:
                    command.accelerator = 0.3
                else:
                    # command.brake = 1.0
                    command.accelerator = 0.6
            else:
                command.accelerator = 1
        else:
            command.accelerator = 0

        if carstate.rpm > 8000:
            command.gear = 2
        else:
            command.gear = 1

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

        return







    ############################################################################
    #
    #  Simple Driver Function for Training
    #
    ############################################################################

    # def simpleDriver(self, carstate, command):
    #
    #     steering = self.steer(carstate, 0.0, command)
    #
    #     ACC_LATERAL_MAX = 6400 * 5
    #     v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
    #     #v_x = 80
    #
    #     acceleration = self.accelerate(carstate, v_x, command)
    #
    #     command_data = [0,0,0] # gas, brake, steering
    #     if acceleration > 0:
    #         command_data[0] = acceleration
    #     else:
    #         command_data[1] = acceleration
    #     command_data[2] = steering
    #
    #     df = pd.DataFrame([command_data+sensor_data,])
    #     #h = ['ACCELERATION','BRAKE','STEERING','SPEED','TRACK_POSITION','ANGLE_TO_TRACK_AXIS','TRACK_EDGE_0','TRACK_EDGE_1','TRACK_EDGE_2','TRACK_EDGE_3','TRACK_EDGE_4','TRACK_EDGE_5','TRACK_EDGE_6','TRACK_EDGE_7','TRACK_EDGE_8','TRACK_EDGE_9','TRACK_EDGE_10','TRACK_EDGE_11','TRACK_EDGE_12','TRACK_EDGE_13','TRACK_EDGE_14','TRACK_EDGE_15','TRACK_EDGE_16','TRACK_EDGE_17','TRACK_EDGE_18']
    #     file_name = './collected_data.csv'
    #     if os.path.isfile(file_name):
    #         df.to_csv(file_name,mode='a',header=False,index=False)
    #     else:
    #         #df.to_csv(file_name,header=h,index=False)
    #         df.to_csv(file_name,header=False,index=False)
    #
    #
    #     return command
    #
    # def accelerate(self, carstate, target_speed, command):
    #     # compensate engine deceleration, but invisible to controller to
    #     # prevent braking:
    #     speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
    #     acceleration = self.acceleration_ctrl.control(
    #         speed_error,
    #         carstate.current_lap_time
    #     )
    #
    #     # stabilize use of gas and brake:
    #     acceleration = math.pow(acceleration, 3)
    #
    #     if acceleration > 0:
    #         if abs(carstate.distance_from_center) >= 1:
    #             # off track, reduced grip:
    #             acceleration = min(0.4, acceleration)
    #
    #         new_acceleration = min(acceleration, 1)
    #         command.accelerator = new_acceleration
    #
    #
    #     else:
    #         new_acceleration = min(-acceleration, 1)
    #         command.brake = new_acceleration
    #
    #
    #     return new_acceleration
    #
    # def steer(self, carstate, target_track_pos, command):
    #     steering_error = target_track_pos - carstate.distance_from_center
    #     new_steering = self.steering_ctrl.control(
    #         steering_error,
    #         carstate.current_lap_time
    #     )
    #     command.steering = new_steering
    #     return new_steering
    #
    # def returnToTrack(self, angle, position):
    #     # returns commands that should bring car back onto the road
    #     # NEEDS WORK
    #     target_angle = np.pi / 4
    #
    #     if position < -1:
    #         if (angle < np.pi / 2) & (angle > -target_angle):
    #             st = 1
    #         else:
    #             st = -1
    #
    #         dif = np.abs(-target_angle - angle)
    #         st *= min(dif, target_angle) / target_angle
    #
    #     else:
    #         if (angle < target_angle) & (angle > -np.pi / 2):
    #             st = -0.6
    #         else:
    #             st = 0.6
    #
    #         dif = np.abs(target_angle - angle)
    #         st *= min(dif, target_angle) / target_angle
    #
    #     # st = np.abs(sensor_TRACK_POSITION) - 1
    #     # st *= - 1.5 * np.sign(sensor_TRACK_POSITION)
    #     ac = 0.2
    #     br = 0.0
    #
    #     return [ac, br, st]
