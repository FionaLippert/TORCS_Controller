import logging
from pytocl.driver import Driver
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import torch
import math
from torch.autograd import Variable



class MyDriver(Driver):

"""
    def drive(self, carstate: State) -> Command:

        command = Command()
        #self.steer(carstate, target_steering, command)
        #self.accelerate(carstate, target_speed, command)

        return command
"""
