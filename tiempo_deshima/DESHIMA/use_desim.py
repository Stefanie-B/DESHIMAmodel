import numpy as np
import math

from .desim import minidesim as dsm
# import DESHIMA.desim.minidesim as dsm

class use_desim(object):

    h = 6.62607004 * 10**-34
    k = 1.38064852 * 10**-23
    e = 1.60217662 * 10**-19 # electron charge
    c = 299792458.
    Delta_Al = 188 * 10**-6 * e # gap energy of Al
    eta_pb = 0.4

    def __init__(self):

        self.instrument_properties_D1 = {
            'eta_M1_spill' : 0.99,
            'eta_M2_spill' : 0.90,
            'n_wo_mirrors' : 2.,
            'window_AR' : False,
            'eta_co' : 0.65, # product of co spillover, qo filter
            'eta_lens_antenna_rad' : 0.81, # D2_2V3.pdf, p14: front-to-back ratio 0.93 * reflection efficiency 0.9 * matching 0.98 * antenna spillover 0.993
            'eta_IBF' : 0.6,
            'KID_excess_noise_factor' : 1.0,
            'Tb_cmb' : 2.725,
            'Tp_amb' : 273.,
            'Tp_cabin' : 290.,
            'Tp_co' : 4.,
            'Tp_chip' : 0.12,
        }

        self.instrument_properties_D2 = {
            'eta_M1_spill' : 0.99,
            'eta_M2_spill' : 0.90,
            'n_wo_mirrors' : 4.,
            'window_AR' : True,
            'eta_co' : 0.65, # product of co spillover, qo filter
            'eta_lens_antenna_rad' : 0.81, # D2_2V3.pdf, p14: front-to-back ratio 0.93 * reflection efficiency 0.9 * matching 0.98 * antenna spillover 0.993
            'eta_IBF' : 0.6,
            'KID_excess_noise_factor' : 1.1,
            'Tb_cmb' : 2.725,
            'Tp_amb' : 273.,
            'Tp_cabin' : 290.,
            'Tp_co' : 4.,
            'Tp_chip' : 0.12,
        }

    def calcLorentzian(F_bins_Lor_mesh, F_filters_mesh, R):
        # F_bins_Lor_mesh, F_filters_mesh = np.meshgrid(F_bins_Lor, F_filters)
        FWHM = F_filters_mesh/R
        y_array = 1/math.pi * 1/2 * FWHM / ((F_bins_Lor_mesh-F_filters_mesh)**2 + (1/2 * FWHM)**2)
        return y_array

    def D2HPBW(F):
        HPBW = 29.*240./(F/1e9) * np.pi / 180. / 60. / 60.
        return HPBW

    def eta_mb_ruze(self, F, LFlimit, sigma):
        '''F in Hz, LFlimit is the eta_mb at => 0 Hz, sigma in m'''
        eta_mb = LFlimit* np.exp(- (4.*np.pi* sigma * F/self.c)**2. )
        return eta_mb

    def transmit_through_DESHIMA(self, signal_instance, pwv_value):
        F_min = signal_instance.F_min
        F_max = signal_instance.F_max
        num_filters = signal_instance.num_filters
        num_bins_Lor = signal_instance.num_bins
        R = signal_instance.spec_res
        eta_atm_df = signal_instance.eta_atm_df
        F_highres = signal_instance.F_highres
        eta_atm_func_zenith = signal_instance.eta_atm_func_zenith
        psd_gal = signal_instance.psd_gal
        EL = signal_instance.EL
        D1 = signal_instance.D1
        pwv_values_no_gal = np.array([pwv_value[0], pwv_value[2], pwv_value[3], pwv_value[4], pwv_value[5]])
        pwv_value_gal = np.array([pwv_value[0], pwv_value[1]])
        F_filters = signal_instance.filters
        margin = 10e9
        # F_bins_Lor = np.logspace(np.log10(F_min-margin), np.log10(F_max + margin), num_bins_Lor)
        F_bins_Lor = np.linspace(F_min-margin, F_max + margin, num_bins_Lor)

        if D1:
            instrument_properties = self.instrument_properties_D1
            theta_maj = 31.4*np.pi/180./60./60.
            theta_min = 22.8*np.pi/180./60./60.
            eta_mb = 0.34
            eta_filter_peak = 0.35 * 0.1
        else:
            instrument_properties = self.instrument_properties_D2
            HPBW = use_desim.D2HPBW(F_bins_Lor)
            eta_mb = self.eta_mb_ruze(F=F_bins_Lor,LFlimit=0.8,sigma=37e-6) * 0.9 # see specs, 0.9 is from EM, ruze is from ASTE
            theta_maj = HPBW
            theta_min = HPBW
            eta_filter_peak = 0.4
        Desim_input_params ={
            'eta_atm_df': eta_atm_df,
            'F_highres': F_highres,
            'eta_atm_func_zenith': eta_atm_func_zenith,
            'F' : F_bins_Lor,
            'pwv':pwv_values_no_gal,
            'EL':EL,
            # 'R' : R,
            'theta_maj' : theta_maj,
            'theta_min' : theta_min,
            'eta_mb' : eta_mb,
            'psd_gal': psd_gal,
            'inclGal': 0
        }
        Desim_input = dict(instrument_properties, **Desim_input_params)
        DESHIMA_transmitted_no_gal = dsm.spectrometer_sensitivity(**Desim_input) # takes a lot of time
        Desim_input_params['pwv'] = pwv_value_gal
        Desim_input_params['inclGal'] = 1
        Desim_input = dict(instrument_properties, **Desim_input_params)
        DESHIMA_transmitted_gal = dsm.spectrometer_sensitivity(**Desim_input) # takes a lot of time
        psd_co_no_gal = DESHIMA_transmitted_no_gal['psd_co'] #vector because of F
        psd_co_gal = DESHIMA_transmitted_gal['psd_co']
        psd_co = np.zeros([num_bins_Lor, 6])
        for i in range(0, 5):
            if i == 0:
                psd_co[:, 0] = psd_co_no_gal[:, 0]
            else:
                psd_co[:, i + 1] = psd_co_no_gal[:, i]
        psd_co[:, 1] = psd_co_gal[:, 1]
        psd_jn_chip = DESHIMA_transmitted_no_gal['psd_jn_chip']
        F_bins_Lor_mesh, F_filters_mesh = np.meshgrid(F_bins_Lor, F_filters)
        eta_circuit = use_desim.calcLorentzian(F_bins_Lor_mesh, F_filters_mesh, R) * math.pi * F_filters_mesh/(2 * R) * eta_filter_peak
        eta_chip = instrument_properties['eta_lens_antenna_rad'] * eta_circuit
        # calculate psd_KID with different values for pwv
        psd_medium = np.transpose((1-eta_chip)*np.transpose(np.array(psd_jn_chip)))
        psd_KID = np.zeros([psd_co.shape[1], num_filters, num_bins_Lor])
        for i in range(num_bins_Lor):
            psd_co_i = psd_co[i, :].reshape(psd_co[i, :].shape[0], 1)
            eta_chip_i = eta_chip[:, i].reshape(1, eta_chip[:, i].shape[0])
            psd_KID_in_i = eta_chip_i*psd_co_i
            result = psd_KID_in_i + psd_medium[i, :]
            psd_KID[:, :, i] = result
        return DESHIMA_transmitted_no_gal, F_bins_Lor, psd_KID, F_filters

##------------------------------------------------------------------------------
## Everything under this is not used in the model, only for making the interpolation curves and plotting
##------------------------------------------------------------------------------

    def obt_data(self, input, D1):
        F = input['F']
        data_names = input['data_names']
        # del(input['data_names'])
        if D1:
            instrument_properties = self.instrument_properties_D1
        else:
            instrument_properties = self.instrument_properties_D2
        sensitivity_input = dict(instrument_properties, **input)
        del(sensitivity_input['data_names'])
        D2goal = dsm.spectrometer_sensitivity(**sensitivity_input) # takes a lot of time
        data = []
        for el in data_names:
            data.append(np.array(D2goal[el]))
        return data

    def calcT_psd_P(self, eta_atm_df, F_highres, eta_atm_func_zenith, F_filter, EL_vector, num_filters, pwv = 0.1, R = 500, num_bins = 1500, D1 = 0):

        length_EL_vector = len(EL_vector)
        margin = 10e9
        # F_bins = np.logspace(np.log10(F_filter[0]-margin), np.log10(F_filter[-1] + margin), num_bins) #to calculate the Lorentzian
        F_bins = np.linspace(F_filter[0] - margin, F_filter[-1] + margin, num_bins)
        if D1:
            instrument_properties = self.instrument_properties_D1
            theta_maj = 31.4*np.pi/180./60./60. * np.ones(num_bins)
            theta_min = 22.8*np.pi/180./60./60. * np.ones(num_bins)
            eta_mb = 0.34 * np.ones(num_bins)
            eta_filter_peak = 0.35 * 0.1
        else:
            instrument_properties = self.instrument_properties_D2
            eta_mb = self.eta_mb_ruze(F=F_bins,LFlimit=0.8,sigma=37e-6) * 0.9 # see specs, 0.9 is from EM, ruze is from ASTE
            HPBW = use_desim.D2HPBW(F_bins)
            theta_maj = HPBW
            theta_min = HPBW
            eta_filter_peak = 0.4
        # Initializing variables
        psd_KID = np.zeros([num_filters, num_bins, length_EL_vector])
        Tb_sky = np.zeros([num_filters, length_EL_vector])
        Pkid = np.zeros([num_filters, length_EL_vector])
        psd_co = np.zeros([num_bins, length_EL_vector])
        psd_jn_chip = np.zeros([num_bins, length_EL_vector])
        eta_circuit = np.zeros(num_bins)

        # Obtain psd_co and psd_jn_chip from desim
        for j in range(0, num_bins):
            input = {
            'F': F_bins[j],
            'pwv': pwv,
            'EL': EL_vector,
            'data_names': ['psd_co', 'psd_jn_chip'],
            'eta_atm_df': eta_atm_df,
            'F_highres': F_highres,
            'eta_atm_func_zenith': eta_atm_func_zenith,
            'theta_maj' : theta_maj[j],
            'theta_min' : theta_min[j],
            'eta_mb' : eta_mb[j]
            }
            [psd_co[j, :], psd_jn_chip[j, :]] = self.obt_data(input, D1)


        # Obtain psd_kid
        for i in range(0, num_filters):
            # Putting a Lorentzian curve with peak height 0.35 and center frequency F_filter[i] in eta_circuit
            eta_circuit = use_desim.calcLorentzian(F_bins, F_filter[i], R) * math.pi * F_filter[i]/(2 * R) * eta_filter_peak
            eta_chip = instrument_properties['eta_lens_antenna_rad'] * eta_circuit
            eta_chip_matrix = np.tile(eta_chip.reshape(len(eta_chip), 1), (1, length_EL_vector))
            psd_KID[i, :, :] =  dsm.rad_trans(psd_co, psd_jn_chip, eta_chip_matrix)


        delta_F = F_bins[1] - F_bins[0]
        numerators = np.zeros([EL_vector.shape[0], num_filters])
        denominators = np.zeros(num_filters)
        for k in range(0, num_filters):
            transmission = use_desim.calcLorentzian(F_bins, F_filter[k], R)
            transmission = transmission.reshape([transmission.shape[0], 1])
            numerators[:, k] =  delta_F * np.sum(transmission \
            * dsm.eta_atm_func(F=F_bins, pwv=pwv, EL=EL_vector, eta_atm_df=eta_atm_df, F_highres=F_highres, eta_atm_func_zenith=eta_atm_func_zenith), axis = 0)
            denominators[k] = delta_F * np.sum(transmission) # delta_F taken out of sum because it is the same for each bin
        eta_atm_matrix = np.transpose(numerators/denominators)

        if D1 == 0:
            eta_mb = self.eta_mb_ruze(F=F_filter,LFlimit=0.8,sigma=37e-6) * 0.9 # see specs, 0.9 is from EM, ruze is from ASTE
            HPBW = use_desim.D2HPBW(F_filter)
            theta_maj = HPBW
            theta_min = HPBW
        # Obtain Tb_sky
        for l in range(0, num_filters):
            input = {
            'F': F_filter[l],
            'pwv': pwv,
            'EL': EL_vector,
            'data_names': ['Tb_sky'],
            'eta_atm_df': eta_atm_df,
            'F_highres': F_highres,
            'eta_atm_func_zenith': eta_atm_func_zenith,
            'eta_atm_smeared': eta_atm_matrix[l, :],
            'theta_maj' : theta_maj[l],
            'theta_min' : theta_min[l],
            'eta_mb' : eta_mb[l]
            }
            Tb_sky[l, :]  = self.obt_data(input, D1)[0]

        return Tb_sky, psd_KID, F_bins
