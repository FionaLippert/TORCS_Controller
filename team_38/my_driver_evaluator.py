#!/usr/bin/python3

import math
from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import neuralNet
from pca import PCA
import numpy as np
import os.path
from sys import stdout
import subprocess
import psutil

class MyDriver(Driver):
    RECOVER_MSG = '+-'*6 + ' RECOVERING ' + '-+'*6

    last_cur_lap_time = 0
    period_end_time = -1
    period_start_dist = 0
    period_distance_raced = 0
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
    train_overtaking = False

    finished_evaluation = True

    def kill(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()

    def __init__(self):

        """
        LAUNCH TORCS WHEN ./start_evolve.sh IS CALLED
        """
        # torcs_command = ["torcs", "-t", "100000", "-nofuel", "-nolaptime", "-nodamage", "-r", "/home/student/Documents/torcs-server/torcs-client/evolver/race_config/quickrace.xml"]
        # self.torcs_process = subprocess.Popen(torcs_command)



    def drive(self, carstate: State):

        # if len(os.listdir('./evolver/kill_torcs/')) > 0:
        #     self.kill(self.torcs_process.pid)

        # evaluation period in seconds
        EVAL_TIME = 90

        t = carstate.current_lap_time
        if t < self.last_cur_lap_time:
            # made a lap, adjust timing events accordingly!
            self.period_end_time -= carstate.last_lap_time
            self.off_time -= carstate.last_lap_time
            self.recovered_time -= carstate.last_lap_time
            self.stopped_time -= carstate.last_lap_time

        self.last_cur_lap_time = t

        use_pca = False

        command = Command()

        '''
        Handeling Evaluation Change:
        '''

        if t > self.period_end_time and not self.recovering:
            # END CURRENT EVALUATION PERIOD:
            if self.first_evaluation:
                # check the right files and folders are available:
                # if not os.path.isdir('./evolver/pool/'):
                #     print('NO POOL FOLDER FOUND')
                #     return command
                #
                # if not os.path.exists('./log.csv'):
                #     print('Creating log.csv...')
                #     open('./log.csv', 'w').close()
                pass
            elif self.esn != None:
                # FITNESS CALCULATION
                # currently average speed (in MPS)
                # penalties for off track and recovery, and using brake too much
                # bonus points for staying in middle of track and keeping small
                # angle

                # self.fitness += (carstate.distance_raced - self.period_start_dist)

                self.fitness += self.period_distance_raced

                # log the performance:
                # save the network with the fitness in the title:
                # new_path = './evolver/queue/%s.pkl'%self.fitness
                # os.system('mv %s %s'%(self.PATH_TO_NEURAL_NET, new_path))

                self.esn.save('./evolver/queue/%.2f.pkl'%self.fitness)

                # with open(new_path, 'wb') as file:
                #     pkl.dump(self,file)

                # nn_id = int(self.PATH_TO_NEURAL_NET[12:-4])
                # log = open('./log.csv', 'a')
                # log.write('%s, %.2f\n'%(nn_id, self.fitness))
                # log.close()
                if self.fitness > 0:
                    print('Fitness Score \033[7m%.2f\033[0m logged\n'%(self.fitness))
                else:
                    print('Fitness Score %.2f logged\n'%(self.fitness))


            # START NEXT EVALUATION PERIOD:
            # load random net:
            pool = os.listdir('./evolver/driver_esn/')
            if len(pool) > 0:
                self.PATH_TO_NEURAL_NET = './evolver/driver_esn/' + str(np.random.choice(pool, 1)[0])
                self.esn = neuralNet.restore_ESN(self.PATH_TO_NEURAL_NET)
                # remove ESN so no other driver can use it:
                os.system('rm %s'%self.PATH_TO_NEURAL_NET)
                print('Evaluating %s:'%self.PATH_TO_NEURAL_NET)

            else:
                # no choice of ESNs, use robot driver
                # print('No ESNS :(')
                self.esn = None
                self.recovering = True

            # reset values:
            self.fitness = 0
            self.period_end_time = t + EVAL_TIME
            self.period_start_dist = carstate.distance_raced
            self.period_distance_raced = 0
            self.first_evaluation = False
            self.init_time = t + 2

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


        # reward fitness for being within 15 degress:
        if np.abs(sensor_ANGLE_TO_TRACK_AXIS) < 0.2618:
            self.fitness += 1

        # reward if in center of road:
        if np.abs(sensor_TRACK_POSITION) < 0.1:
            self.fitness += 1


        """
        Process Inputs.

        Feed the sensor data into the network to produce a Accelrator, Brake
        and Steering command
        """

        sensor_data = [min(sensor_SPEED, 300)/300, sensor_TRACK_POSITION, sensor_ANGLE_TO_TRACK_AXIS]

        if self.is_stopped & (t > self.stopped_time + 3) & (not self.recovering):

            self.recovering = True
            self.is_stopped = False
            # print(self.RECOVER_MSG)
            # print('Stopped for 3 seconds...')
            self.fitness -= 5000
            self.finished_evaluation = True


        if self.recovering:
            self.simpleDriver(sensor_data, carstate, command)

            if np.abs(sensor_TRACK_POSITION) < 1:
                # recovered!
                # self.stuck = 0
                if not self.warm_up:
                    # print('Back on track...')
                    self.recovered_time = t
                    self.warm_up = True

                self.off_track = False

                # considered recovered if moving fast and straightish
                if (t > self.recovered_time + 5) & (sensor_SPEED > 40) & (np.abs(sensor_ANGLE_TO_TRACK_AXIS) < 0.5):
                    self.recovering = False
                    self.warm_up = False
                    self.period_end_time = t  # will end evaluation period
                    # print('Recovered and starting new evaulation')
                    # print('+-'*18)

            else:
                self.off_track = True
                self.warm_up = False

        else:

            '''
            Drive using Neural Net
            '''

            if np.abs(sensor_TRACK_POSITION) > 1:
                if self.off_track == False:
                    # print("### OFF ROAD ###")

                    self.off_time = t

                self.off_track = True
                self.fitness -= 1

                if t > self.off_time + 5:
                    # haven't recovered in 3 seconds
                    # get back on road and start new evaluation
                    self.fitness -= 5000   # penalty
                    self.recovering = True
                    self.finished_evaluation = True
                    # print(self.RECOVER_MSG)

            else:
                self.off_track = False
                self.period_distance_raced = carstate.distance_raced - self.period_start_dist

            x = np.array(sensor_TRACK_EDGES) / 200
            if use_pca:
                y = PCA.convert(x.T)
                sensor_data += list(y)
            else:
                sensor_data += list(x)

            # use EchoStateNet

            try:
                output = self.esn.predict(sensor_data,continuation=True)
            except:
                # pass
                self.esn = neuralNet.restore_ESN(self.PATH_TO_NEURAL_NET)
                output = self.esn.predict(sensor_data,continuation=True)
                # print('Loaded ' + self.PATH_TO_NEURAL_NET)


            if output[0] > 0:
                if sensor_SPEED < 120:
                    accel = min(max(output[0],0),1)
                else:
                    accel = 0
                brake = 0.0
            else:
                accel = 0.0
                brake = min(max(-output[0],0),1)

                if sensor_SPEED < 10:
                    self.fitness -= 1
                else:
                    if np.abs(sensor_ANGLE_TO_TRACK_AXIS) > 0.2618:
                        self.fitness += 1

            steer = min(max(output[1],-1),1)

            # if np.abs(steer) < 0.1 and x[9] > 0.9:
            #     self.fitness += 1
            #
            # if np.abs(steer) > 0.8 and x[9] < 0.2:
            #     self.fitness += 1



            """
            Apply Accelrator, Brake and Steering Commands from the neural net
            """

            if self.train_overtaking:
                # reduce range to within 10m
                opponents = np.array(carstate.opponents)
                opponents = np.minimum(10, opponents)

                if min(opponents) < 10:
                    # opponent nearby, engage overtaking network
                    input2 = [output[0], steer]
                    input2 += list(opponents / 10)

                    ############################################
                    # CODE TO ALTER OUTPUTS
                    ############################################




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

        return command






    ############################################################################
    #
    #  Simple Driver Function for Recovery
    #
    ############################################################################
    stuck = 0

    def isStuck(self, angle, carstate):

        if (carstate.speed_x < 3) & (np.abs(carstate.distance_from_center) > 0.7) & (np.abs(angle) > 20/180*np.pi):
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
