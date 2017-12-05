#!/usr/bin/python3

import math
from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import neuralNet
#import pandas as pd
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



    """
    overwrite the constructor of the 'Driver' class
    """

    def __init__(self, logdata=False):


        """
        set the initialization mode
        """
        self.launch_torcs = False
        self.launch_torcs_and_second_driver = True

        """
        these parameters determine which controllers to use

        use_simple_driver: if True, a simple hard coded driving logic is applied
                            else, the ESN saved at PATH_TO_ESN is used to predict the commands
        load_w_out_from_file: if True, the readout weights of the ESN are overwritten by those saved at PATH_TO_W_OUT
        use_mlp_opponents: if True, the output of the ESN is adjusted by the Mulit-Layer Perceptron saved at PATH_TO_MLP
        """
        self.use_simple_driver = False
        self.load_w_out_from_file = False
        self.use_mlp_opponents = True

        self.PATH_TO_ESN = "./trained_nn/esn.pkl"
        self.PATH_TO_W_OUT = "./w_out.npy"
        self.PATH_TO_MLP = "./trained_nn/mlp_opponents.pkl"




        """
        Controllers needed for the simple driver
        """
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

        # automatically start TORCS when './start.sh' is executed
        if self.launch_torcs:
            torcs_command = ["torcs","-r",os.path.abspath("./config_files/current_config_file/config.xml")]
            self.torcs_process = subprocess.Popen(torcs_command)

        # automatically start TORCS with two drivers when './start.sh' is executed
        elif self.launch_torcs_and_second_driver:

            # assure that TORCS is not launched again when the second car is initialized
            
            if not os.path.isfile('torcs_process.txt'):
                torcs_command = ["torcs","-r",os.path.abspath("./config_files/current_config_file/config.xml")]
                self.torcs_process = subprocess.Popen(torcs_command)

                with open('./torcs_process.txt', 'w') as file:
                    file.write('running torcs process')

                second_car_command = ["./start.sh","-p","3002"]
                subprocess.Popen(second_car_command)
            else:
                line = ''
                with open('./torcs_process.txt', 'r') as file:
                    line = file.readline()
                if not line.find('running torcs process') > -1:

                    torcs_command = ["torcs","-r",os.path.abspath("./config_files/current_config_file/config.xml")]
                    self.torcs_process = subprocess.Popen(torcs_command)

                    with open('./torcs_process.txt', 'w') as file:
                        file.write('running torcs process')

                    second_car_command = ["./start.sh","-p","3002"]
                    subprocess.Popen(second_car_command)
                else:
                    with open('./torcs_process.txt', 'w') as file:
                        file.write('')



        """
        some global variables
        """

        self.previous_position = 0
        self.active_mlp = False
        self.current_damage = 0
        self.accel_deviation = 0
        self.steering_deviation = 0


    """
    method that determines the driving commands to be passed on to the TORCS server
    (overwrites the drive function the 'Driver' class)
    """

    def drive(self, carstate: State):

        # at the begin of the race: determine on which position the car starts
        if self.previous_position == 0:
            self.previous_position = carstate.race_position

        t = carstate.current_lap_time
        if t < self.last_cur_lap_time:
            # made a lap, adjust timing events accordingly!
            self.period_end_time -= carstate.last_lap_time
            self.off_time -= carstate.last_lap_time
            self.recovered_time -= carstate.last_lap_time
            self.stopped_time -= carstate.last_lap_time

        self.last_cur_lap_time = t


        # initialize the driving command
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
        x = np.array(sensor_TRACK_EDGES) / 200
        sensor_data += list(x)



        if self.is_stopped & (t > self.stopped_time + 3) & (not self.recovering):
            self.recovering = True
            self.is_stopped = False
            #print(self.RECOVER_MSG)
            #print('Stopped for 3 seconds...')

            with open('./simulation_log.txt', 'a') as file:
                file.write('stopped for 3 seconds\n')



        if self.recovering:

            """
            Recovery Bot
            """

            self.simpleDriver(sensor_data, carstate, command)

            if np.abs(sensor_TRACK_POSITION) < 1:
                # recovered!
                # self.stuck = 0
                if not self.warm_up:
                    #print('Back on track...')
                    self.recovered_time = t
                    self.warm_up = True

                self.off_track = False

                # considered recovered if moving fast and straightish
                if (t > self.recovered_time + 5) & (sensor_SPEED > 40) & (np.abs(sensor_ANGLE_TO_TRACK_AXIS) < 0.5):
                    self.recovering = False
                    self.warm_up = False
                    self.period_end_time = t  # will end evaluation period
                    #print('Recovered and starting new evaulation')
                    #print('+-'*18)

            else:
                self.off_track = True
                self.warm_up = False
        else:

            """
            Drive using the EchoStateNet
            """

            # check if car is off road and if so, for how long
            if np.abs(sensor_TRACK_POSITION) > 1:
                if self.off_track == False:
                    #print("### OFF ROAD ###")
                    with open('./simulation_log.txt', 'a') as file:
                        file.write('driver %.0f off road\n'%carstate.race_position)
                    self.off_time = t

                self.off_track = True

                if t > self.off_time + 3:
                    # haven't recovered in 3 seconds
                    # get back on road and start new evaluation
                    self.fitness -= 100   # penalty
                    self.recovering = True
                    #print(self.RECOVER_MSG)

            else:
                self.off_track = False
                with open('./simulation_log.txt', 'a') as file:
                    file.write('driver %.0f on road\n'%carstate.race_position)



            """
            load ESN if it is not yet available,
            predict the driving commands
            """
            try:
                output = self.esn.predict(sensor_data,continuation=True)
                # print('esn already loaded')
            except:
                # load ESN from specified path
                self.esn = neuralNet.restore_ESN(self.PATH_TO_ESN)

                # if desired, set the ESN readout weights to externally given ones
                if self.load_w_out_from_file:
                    self.esn.load_w_out(self.PATH_TO_W_OUT)

                # start a new 'history of states'
                output = self.esn.predict(sensor_data,continuation=False)


            """
            Controller extension: Multi-Layer Perceptron that adjust the driving commands if opponents are close
            """

            # modifiy ESN output based on the opponents data
            # only if car is not racing at the first position
            self.active_mlp = False

            if self.use_mlp_opponents and carstate.race_position > 1:
                opponents_data = carstate.opponents


                # if closest opponent is less than 10m away, use mlp to adjust outputs
                if min(opponents_data) < 50:

                    self.active_mlp = True
                    #print('-------------use mlp---------------')

                    mlp_input = [output[0],output[1]]
                    #print(opponents_data)
                    for sensor in opponents_data:
                        # normalize opponents_data to [0,1]
                        mlp_input.append(sensor/10.0)

                    """
                    load MLP if it is not yet available,
                    predict the driving commands
                    """
                    try:
                        output = self.mlp.predict(np.asarray(mlp_input))
                    except:
                        self.mlp = neuralNet.restore_MLP(self.PATH_TO_MLP)
                        output = self.mlp.predict(np.asarray(mlp_input))

                    # determine the deviation between ESN and MLP output
                    self.accel_deviation = abs(output[0]-mlp_input[0])
                    self.steering_deviation = abs(output[1]-mlp_input[1])


            """
            determine if car should accelerate or brake
            """

            if output[0] > 0:
                accel = min(max(output[0],0),1)
                brake = 0.0
            else:
                accel = 0.0
                brake = min(max(-output[0],0),1)

            steer = min(max(output[1],-1),1)

            # uncomment to print current input and output data
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
        write data for evaluation to the file 'simulation_log.txt'
        """

        with open('./simulation_log.txt', 'a') as file:

            file.write('----------------car at position %.0f----------------------\n'%carstate.race_position)
            file.write('current_lap_time: '+str(carstate.current_lap_time)+'\n')
            if carstate.distance_from_start <= carstate.distance_raced:
                file.write('distance_from_start: '+str(carstate.distance_from_start)+'\n')
            else:
                file.write('distance_from_start: 0 \n')

            # if overtaking was successful
            if self.active_mlp and carstate.race_position > self.previous_position:
                file.write('overtaking successful \n')

            #file.write('damage: '+str(carstate.damage)+'\n')
            file.write('distance from center: '+str(carstate.distance_from_center)+'\n')

            file.write('distance to opponent: '+str(min(carstate.opponents))+'\n')

            file.write('angle: '+str(carstate.angle)+'\n')

            if self.active_mlp and self.current_damage<carstate.damage:
                file.write('MLP damage \n')
            if self.active_mlp:
                file.write('MLP d from center: '+str(carstate.distance_from_center)+'\n')
                file.write('MLP accelerator deviation: '+str(self.accel_deviation)+'\n')
                file.write('MLP steering deviation: '+str(self.steering_deviation)+'\n')

        # save damage and position information for the next simulation step
        self.current_damage = carstate.damage
        self.previous_position = carstate.race_position

        return command









    """
    Simple Driver Function for Recovery
    """

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


    def simpleDriver(self, sensor_data, carstate, command):

        if self.isStuck(sensor_data[2], carstate):
            command.gear = -1
            command.steering = np.sign(sensor_data[1]) * 0.6
            command.brake = 0
            command.accelerator = 0.5


        else:
            if self.off_track:
                self.returnToTrack(sensor_data[2], sensor_data[1], command)

            else:
                # Drive normally
                self.steer(carstate, 0.0, command)

                ACC_LATERAL_MAX = 6400 * 5
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