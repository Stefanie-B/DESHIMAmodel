import math
import numpy as np


class use_ARIS(object):
    """
    This class is used to convert the atmosphere data that is obtained from
    ARIS to a useful datatype. It loads in the atmosphere data, converts it to
    a matrix and converts the Extra Path Length to the precipitable water vapor
    using the Smith-Weintraub equation for the Extra Path Length and the Ideal
    Gas Law (later, hopefully the VanDerWaals law).
    """

    a = 6.3003663 #m
    h = 1000 #m

    def __init__(self, x_length_strip, sourcepath, prefix_filename, pwv_0, grid, windspeed, time, max_num_strips, separation, beam_radius, loadCompletePwvMap = 0):
        #make function to convert EPL to pwv
        self.pwv_0 = pwv_0
        self.prefix_filename = prefix_filename
        self.grid = grid
        self.windspeed = windspeed
        self.max_num_strips = max_num_strips
        self.sourcepath = sourcepath
        self.x_length_strip = x_length_strip
        self.filtered_pwv_matrix = "None"
        self.separation = separation
        self.beam_radius = beam_radius
        if loadCompletePwvMap:
            self.load_complete_pwv_map()
        else:
            self.initialize_pwv_matrix(time)

    def calc_e_from_pwv(pwv):
        """Calculates the simple partial pressure of water vapor, using the
        VanDerWaals equation of state and values for the parameters a and b
        taken from 'CRC Handbook for chemistry and physics', 64th edition.

        Returns
        ------------
        e: scalar or array
            partial pressure of water vapor
            Unit: kPa
        """
        a = 5.536e-4
        b = 3.049e-5
        T = 273
        rho = 55.4e3
        R = 8.314459
        e = R * T * pwv/(use_ARIS.h - b * pwv * rho) - (a * (pwv**2) * (rho**2)) / (use_ARIS.h**2)
        return e

    def calc_e_from_EPL(self, EPL):
        """Calculates the partial pressure of water vapor, using the Smith-Weintraub
        formula for Extra Path Length (DOI: 10.1109/JRPROC.1953.274297)

        Returns
        ------------
        e: scalar or array
            partial pressure of water vapor
            Unit: kPa
        """
        k2 = 70.4e2 #K/kPa
        k3 = 3.739e7 #K**2/kPa
        T = 273
        e = 1e6/self.h * EPL/(k2/T + k3/(T**2))
        return e
    """
    def load_complete_pwv_map(self):
        """"""Loads the previously saved pwv map. This map is already filtered with
        a Gaussian beam.
        """"""
        print('Number of atmosphere strips loaded: ', self.max_num_strips)
        #path = self.path_model + '/Data/output_ARIS/complete_filtered_pwv_map.txt'
        path = self.sourcepath.joinpath('complete_filtered_pwv_map.txt')
        self.filtered_pwv_matrix = np.loadtxt(path)
        """
    def initialize_pwv_matrix(self, time):
        """Initializes the pwv_matrix property of the use_ARIS instance. It loads
        in the amount of atmosphere strips that it needs, converts it into a matrix
        and then converts EPL to pwv.
        """
        max_distance = (time*self.windspeed + 2*self.separation)
        max_x_index = math.ceil(max_distance/self.grid)
        num_strips = min(math.ceil(max_x_index/self.x_length_strip), self.max_num_strips)
        grid_dif = int(round(self.separation/self.grid))
        dEPL_size = int(round(self.beam_radius/self.grid))+2*grid_dif
        #print('Number of atmosphere strips loaded: ', num_strips)
        for i in range(num_strips):
            filename = self.prefix_filename + (3-len(str(i))) * "0" + str(i)
            d = np.loadtxt(self.sourcepath.joinpath(filename), dtype = float, delimiter=',', skiprows = 18)
            if i == 0:
                nx = int(max(d[:, 0])) + 1
                ny = int(max(d[:, 1])) + 1
                # print('nx: ', nx)
                # print('ny: ', ny)
            epl= np.zeros([nx,ny])
            for j in range(len(d)):
                epl[int(d[j, 0])-int(d[0, 0]), int(d[j, 1])] = int(d[j, 2])
            if i == 0:
                self.dEPL_matrix = epl[:, 0:dEPL_size]
            else:
                self.dEPL_matrix = np.concatenate((self.dEPL_matrix, epl[:, 0:dEPL_size]), axis = 0)
        self.pwv_matrix = self.pwv_0 + (1/self.a * self.dEPL_matrix*1e-6)*1e3 #in mm
        self.pwv_matrix = np.array(self.pwv_matrix)

    def obt_pwv(self, time, count, windspeed):
        """Obtains the precipitable Water vapor from the pwv_matrix at 5 different positions, to
        make sky chopping and nodding possible in 2 directons.

        Returns
        ------------
        pwv: array
            precipitable water vapor on at 5 different positions.
            Unit: m
        """
        pwv_matrix = self.filtered_pwv_matrix
        positions = self.calc_coordinates(time, windspeed)
        pwv = np.array([pwv_matrix[positions[0]], pwv_matrix[positions[1]], pwv_matrix[positions[2]], pwv_matrix[positions[3]], pwv_matrix[positions[4]], pwv_matrix[positions[5]]])
        return pwv

    def calc_coordinates(self, time, windspeed):
        """Calculates the positions at which the pwv needs to be taken. Five
        different positions are taken, to make sky chopping and nodding possible
        in 2 directons.

        Returns
        ------------
        Positions: array
            The positions (gridpoints) at which the pwv needs to be taken.
        """
        grid_dif = int(round(self.separation/self.grid)) #number of gridpoints difference between positions
        distance = time*windspeed
        x_index = (int(round(distance/self.grid)))
        y_index = int(round(self.beam_radius/self.grid))+grid_dif - 1#int(23/self.grid)-1 #23m, the -1 is to be in accordance with python array indexing
        pos_1 = x_index + grid_dif, y_index
        pos_2 = x_index + grid_dif, y_index
        pos_3 = x_index + grid_dif, y_index
        pos_4 = x_index + grid_dif, y_index + grid_dif
        pos_5 = x_index + grid_dif, y_index - grid_dif
        pos_6 = x_index + grid_dif, y_index
        positions = [pos_1, pos_2, pos_3, pos_4, pos_5, pos_6]
        return positions
