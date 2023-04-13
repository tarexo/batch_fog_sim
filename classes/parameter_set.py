import math
import numpy as np
from scipy.constants import speed_of_light as c


class ParameterSet:
    def __init__(self, **kwargs) -> None:
        self.n = 500
        self.n_min = 100
        self.n_max = 1000

        self.r_range = 100
        self.r_range_min = 50
        self.r_range_max = 250

        ##########################
        # soft target a.k.a. fog #
        ##########################

        # attenuation coefficient => amount of fog
        self.alpha = 0.06
        self.alpha_min = 0.003
        self.alpha_max = 0.5
        self.alpha_scale = 1000

        # meteorological optical range (in m)
        self.mor = np.log(20) / self.alpha

        # backscattering coefficient (in 1/sr) [sr = steradian]
        self.beta = 0.046 / self.mor
        self.beta_min = 0.023 / self.mor
        self.beta_max = 0.092 / self.mor
        self.beta_scale = 1000 * self.mor

        ##########
        # sensor #
        ##########

        # pulse peak power (in W)
        self.p_0 = 80
        self.p_0_min = 60
        self.p_0_max = 100

        # half-power pulse width (in s)
        self.tau_h = 2e-8
        self.tau_h_min = 5e-9
        self.tau_h_max = 8e-8
        self.tau_h_scale = 1e9

        # total pulse energy (in J)
        self.e_p = self.p_0 * self.tau_h  # equation (7) in [1]

        # aperture area of the receiver (in in m²)
        self.a_r = 0.25
        self.a_r_min = 0.01
        self.a_r_max = 0.1
        self.a_r_scale = 1000

        # loss of the receiver's optics
        self.l_r = 0.05
        self.l_r_min = 0.01
        self.l_r_max = 0.10
        self.l_r_scale = 100

        self.c_a = c * self.l_r * self.a_r / 2

        self.linear_xsi = True

        self.D = 0.1  # in m              (displacement of transmitter and receiver)
        self.ROH_T = 0.01  # in m              (radius of the transmitter aperture)
        self.ROH_R = 0.01  # in m              (radius of the receiver aperture)
        self.GAMMA_T_DEG = (
            2  # in deg            (opening angle of the transmitter's FOV)
        )
        self.GAMMA_R_DEG = (
            3.5  # in deg            (opening angle of the receiver's FOV)
        )
        self.GAMMA_T = math.radians(self.GAMMA_T_DEG)
        self.GAMMA_R = math.radians(self.GAMMA_R_DEG)

        # assert self.GAMMA_T_DEG != self.GAMMA_R_DEG, 'would lead to a division by zero in the calculation of R_2'
        #
        # self.R_1 = (self.D - self.ROH_T - self.ROH_R) / (
        #         np.tan(self.GAMMA_T / 2) + np.tan(self.GAMMA_R / 2))  # in m (see Figure 2 and Equation (11) in [1])
        # self.R_2 = (self.D - self.ROH_R + self.ROH_T) / (
        #         np.tan(self.GAMMA_R / 2) - np.tan(self.GAMMA_T / 2))  # in m (see Figure 2 and Equation (12) in [1])

        # R_2 < 10m in most sensors systems
        # co-axial setup (where R_2 = 0) is most affected by water droplet returns

        # range at which receiver FOV starts to cover transmitted beam (in m)
        self.r_1 = 0.9
        self.r_1_min = 0
        self.r_1_max = 10
        self.r_1_scale = 10

        # range at which receiver FOV fully covers transmitted beam (in m)
        self.r_2 = 1.0
        self.r_2_min = 0
        self.r_2_max = 10
        self.r_2_scale = 10

        ###############
        # hard target #
        ###############

        # distance to hard target (in m)
        self.r_0 = 30
        self.r_0_min = 1
        self.r_0_max = 200

        # reflectivity of the hard target [0.07, 0.2, > 4 => low, normal, high]
        self.gamma = 0.000001
        self.gamma_min = 0.0000001
        self.gamma_max = 0.00001
        self.gamma_scale = 10000000

        # differential reflectivity of the target
        self.beta_0 = self.gamma / np.pi

        self.__dict__.update(kwargs)

    def set_params(
        self, alpha: float = 0.06, beta: float = 0.046, gamma: float = 0.000001
    ):
        self.alpha = alpha
        self.mor = np.log(20) / self.alpha
        self.beta = beta / self.mor
        self.gamma = gamma
        self.beta_0 = self.gamma / np.pi
