"""
DEshima SIMulator.
Module for calculating the sensitivity of a DESHIMA-type spectrometer.

FAQ
-----
Q.  Where is the point-source coupling phase and amplitude loss
    due to the mismatch between the
    beam in radiation and reception, that Shahab calculates
    at the lens surface?
        A.  It is included in the main beam efficiency.
            These losses reduce the coupling to a point source,
            but the power (in transmission) couples to the sky.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
import copy
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages
#from IPython.display import HTML
import base64

h = 6.62607004 * 10**-34  # Planck constant
k = 1.38064852 * 10**-23  # Boltzmann constant
e = 1.60217662 * 10**-19  # electron charge
c = 299792458.  # velocity of light
Delta_Al = 188. * 10**-6 * e  # gap energy of Al
eta_pb = 0.4  # Pair breaking efficiency
eta_Al_ohmic_850 = 0.9975 # Ohmic loss of an Al surface at 850 GHz. Shitov et al., ISSTT2008. https://www.nrao.edu/meetings/isstt/papers/2008/2008263266.pdf

def spectrometer_sensitivity(
        eta_atm_df: pd.Series([]),
        F_highres: pd.Series([]),
        eta_atm_func_zenith: pd.Series([]),
        eta_atm_smeared=0,
        F=350.*10.**9.,  # Hz, scalar or vector
        pwv=0.5,  # mm, scalar
        EL=60.,  # deg, scalar
        # R=500,  # scalar. R = Q = F/FWHM = F/dF is assumed.
        eta_M1_spill=0.99,  # scalar or vector
        eta_M2_spill=0.90,  # scalar or vector
        eta_wo_spill=0.99,  # scalar or vector. product of cabin spillover loss
        n_wo_mirrors = 4., # number of cabin optics, excluing telescope M1 and M2
        window_AR=True,
        # scalar or vector. product of co spillover, qo filter transmission
        eta_co=0.65,
        # scalar or vector. D2_2V3.pdf, p14:
        # front-to-back ratio 0.93 * reflection efficiency 0.9
        # * matching 0.98 * antenna spillover 0.993
        eta_lens_antenna_rad=0.81,
        # scalar or vector. 'Alejandro Efficiency',
        # from the feedpoint of the antenna to being absorbed in the KID.
        # eta_circuit=0.35,
        eta_IBF=0.6,
        KID_excess_noise_factor = 1.1,
        theta_maj=22. * np.pi / 180. / 60. / 60.,  # scalar or vector.
        theta_min=22. * np.pi / 180. / 60. / 60.,  # scalar or vector.
        eta_mb=0.6,  # scalar or vector.
        telescope_diameter=10.,
        Tb_cmb=2.725,
        Tp_amb=273.,
        Tp_cabin=290.,
        Tp_co=4.,
        Tp_chip=0.12,
        snr=5.,
        obs_hours=10.,
        on_source_fraction=0.4*0.9,
        psd_gal = 0,
        inclGal = 0,
        on_off = True
        ):
    """
    Calculate the sensitivity of a spectrometer.

    Parameters
    ------------
    One of the parameters can be a vector of length N.
    In this case the output of the script will be a 2D pandas.DataFrame
    with N rows.
    If all parameters are a scalar, then the output will be
    a 1D pandas DataFrame (or Series?)

    F : scalar or vector
        Frequency of the astronomical signal.
        Unit: Hz
    pwv : scalar or vector
        precipitable water vapour.
        Unit: mm
    EL : scalar or vector
        telescope elevation angle
        Unit: degrees
    R : scalar or vector
        spectral resolving power in F/W_F
        W_F is the 'equivalent bandwidth'
            http://www.astrosurf.com/buil/us/spe2/hresol7.htm
        Unit : None.
    eta_M1_spill : scalar or vector
        spillover efficiency at the telescope primary mirror.
        Unit: None.
    eta_M2_spill : scalar or vector
        spillover efficiency at the telescope secondary mirror.
        Unit: None.
    eta_wo_spill : scalar or vector
        Product of all spillover losses in the warm optics in the cabin.
        Unit: None.
    eta_co : scalar or vector
        Product of following:
            - Cold spillover
            - Cold ohmic losses
            - Filter transmission loss
        Unit: None.
    eta_lens_antenna_rad : scalar or vector
        The loss at chip temperature, *that is not in the circuit.*
        Product of the following:
            - Front-to-back ratio of the lens-antenna on the chip.
            - Reflection efficiency at the surface of the lens.
            - Matching efficiency, due to the mismatch
            - Spillover efficiency of the lens-antenna
        These values can be found in D2_2V3.pdf, p14
        Unit: None.
    eta_circuit : scalar or vector
        The loss at chip temperature, *in the circuit.*
        Unit: None.
    eta_IBF : in-band fraction efficiency
        Fraction of the filter power transmission that is
        within the filter channel bandwidth.
        The rest of the power is cross talk,
        picking up power that is in the bands of neighboring channels.
        This efficiency applies to the coupling to astronomical line signals.
        This efficiency does not apply to the coupling to continuum,
        including the the coupling to the atmosphere for calculating the NEP.
    theta_maj : scalar or vector
        The HPBW along the major axis, assuming a Gaussian beam.
        Unit: radians.
    theta_min : scalar or vector
        The HPBW along the minor axis, assuming a Gaussian beam.
        Unit: radians.
    eta_mb : scalar or vector
        main beam efficiency.
        Note that eta_mb includes the following terms from D2_2V3.pdf
        from Shahab's report.
            - eta_Phi
            - eta_amp
        Because a decrease in these will launch the beam to the sky b
        ut not couple it to the point source.
        (See also FAQ.)
        Unit: None.
    telescope_diameter : float or vector.
        diameter of the telescope.
        Units: m
    Tb_cmb : scalar or vector
        Brightness temperature of the CMB.
        Unit: K
    Tp_amb : scalar or vector
        Physical temperature of the atmosphere and ambient environment
        around the telescope.
        Unit: K
    Tp_cabin : scalar or vector
        Physical temperature of the telescope cabin.
        Unit: K
    Tp_co : scalar or vector
        Physical temperature of the cold optics inside the cryostat.
        Unit: K
    Tp_chip : scalar or vector
        Physical temperature of the chip.
        Unit: K
    snr : scalar or vector
        target signal to noise to be reached. (for calculating the MDLF)
        Unit: None.
    obs_hours :  scalar or vector
        observing hours, including off-source time and the slew
        overhead between on- and off-source.
        Unit : hours
    on_source_fraction : scalar or vector
        Fraction of the time on source (between 0. and 1.)
    on_off: True or False
        If the observation involves on_off chopping, then the SNR degrades by sqrt(2) because
        the signal difference includes the noise twice.
    psd_gal: vector
        Power spectral density of the galaxy that is used in the simulation.
        Unit: W/Hz
    inclGal: integer
        Equals 1 if the galaxy spectrum needs to be added. Equals 0 otherwise

    Returns
    ----------
    The function returns one python dictionary called 'result',
    which contains the following keys
    F : same as input
    eta_atm : atmospheric transmission. Unit: None
    R : same as input
    Tb_sky : Planck brightness temperature of the sky. Units : K
    psd_co: Johnson-Nyquist power spectral density of the cold-optics temperature. Units: W/Hz
    psd_jn_chip: Johnson-Nyquist power spectral density of the chip. Units: W/Hz
    psd_KID: Johnson-Nyquist power spectral density of the KID (Kinetic Inductance Detector). Units: W/Hz

    Note
    ----------------
    - The parameters to calculate the window transmission / reflection
    is hard-coded in the function window_trans().
    """
    # #############################################################
    # 1. Calculating loading power absorbed by the KID, and the NEP
    # #############################################################

    # .......................................................
    # Efficiencies for calculating sky coupling
    # .......................................................

    # Ohmic loss as a function of frequency, from skin effect scaling
    eta_Al_ohmic = (1.-(1.-eta_Al_ohmic_850)*np.sqrt(F/850.e9))
    if type(eta_Al_ohmic) != np.float64:
        eta_Al_ohmic = eta_Al_ohmic.reshape([eta_Al_ohmic.shape[0], 1])
    eta_M1_ohmic = eta_Al_ohmic
    eta_M2_ohmic = eta_Al_ohmic

    # Collect efficiencies at the same temperature
    eta_M1 = eta_M1_ohmic * eta_M1_spill
    eta_wo = eta_Al_ohmic**n_wo_mirrors * eta_wo_spill
    # eta_chip = eta_lens_antenna_rad * eta_circuit

    if type(eta_atm_smeared) == int:
        # Calcuate eta. scalar/vector depending on F.
        eta_atm = eta_atm_func(F=F, pwv=pwv, EL=EL, eta_atm_df=eta_atm_df, F_highres=F_highres, eta_atm_func_zenith=eta_atm_func_zenith)
    else:
        eta_atm = eta_atm_smeared

    # Johnson-Nyquist Power Spectral Density (W/Hz) for the physical temperatures of each stage
    if inclGal == 1:
        psd_jn_cmb_and_gal   = johnson_nyquist_psd(F=F, T=Tb_cmb) + psd_gal
    else:
        psd_jn_cmb_and_gal   = johnson_nyquist_psd(F=F, T=Tb_cmb)
    if type(psd_jn_cmb_and_gal) != np.float64:
        psd_jn_cmb_and_gal = psd_jn_cmb_and_gal.reshape([psd_jn_cmb_and_gal.shape[0], 1])
    psd_jn_amb   = johnson_nyquist_psd(F=F, T=Tp_amb)
    if type(psd_jn_amb) != np.float64:
        psd_jn_amb = psd_jn_amb.reshape([psd_jn_amb.shape[0], 1])
    psd_jn_cabin = johnson_nyquist_psd(F=F, T=Tp_cabin)
    if type(psd_jn_cabin) != np.float64:
        psd_jn_cabin = psd_jn_cabin.reshape([psd_jn_cabin.shape[0], 1])
    psd_jn_co    = johnson_nyquist_psd(F=F, T=Tp_co)
    if type(psd_jn_co) != np.float64:
        psd_jn_co = psd_jn_co.reshape([psd_jn_co.shape[0], 1])
    psd_jn_chip  = johnson_nyquist_psd(F=F, T=Tp_chip)
    if type(psd_jn_chip) != np.float64:
        psd_jn_chip = psd_jn_chip.reshape([psd_jn_chip.shape[0], 1])

    # Optical Chain
    # Sequentially calculate the Power Spectral Density (W/Hz) at each stage.
    # Uses only basic radiation transfer: rad_out = eta*rad_in + (1-eta)*medium
    psd_sky =       rad_trans(rad_in=psd_jn_cmb_and_gal,   medium=psd_jn_amb,     eta=eta_atm     )
    psd_M1  =       rad_trans(rad_in=psd_sky,      medium=psd_jn_amb,     eta=eta_M1      )
    psd_M2_spill =  rad_trans(rad_in=psd_M1,       medium=psd_sky,        eta=eta_M2_spill) # Note that the spillover of M2 is to the
    psd_M2 =        rad_trans(rad_in=psd_M2_spill, medium=psd_jn_amb,     eta=eta_M2_ohmic)
    psd_wo =        rad_trans(rad_in=psd_M2,       medium=psd_jn_cabin,   eta=eta_wo      )
    [psd_window, eta_window] = (
                    window_trans(F=F, psd_in=psd_wo, psd_cabin=psd_jn_cabin, psd_co=psd_jn_co, window_AR=window_AR))
    psd_co =        rad_trans(rad_in=psd_window,   medium=psd_jn_co,      eta=eta_co      )
    # psd_KID =       rad_trans(rad_in=psd_co,       medium=psd_jn_chip,    eta=eta_chip    )  # PSD absorbed by KID

    Tb_sky = T_from_psd(F, psd_sky)

    # ############################################
    # 3. Output results as dictionary
    # ############################################

    result = {
        'F': F,
        'eta_atm': eta_atm,
        'psd_co': psd_co,
        'psd_jn_chip': psd_jn_chip,
        # 'psd_KID': psd_KID,
        'Tb_sky': Tb_sky
    }
    return result

def calc_eff_aper(F, beam_radius):
    # Galaxy spectrum
    c = 299792458.
    eta_mb = eta_mb_ruze(F=F,LFlimit=0.8,sigma=37e-6) * 0.9 # expects F in Hz, see specs, 0.9 is from EM, ruze is from ASTE
    theta_maj = D2HPBW(F)
    theta_min = D2HPBW(F)
    Ag = np.pi * (beam_radius)**2.  # Geometric area of the telescope
    omega_mb = np.pi * theta_maj * theta_min / np.log(2) / 4 # Main beam solid angle
    omega_a = omega_mb / eta_mb # beam solid angle
    Ae = (c/F)**2 / omega_a # Effective Aperture (m^2): lambda^2 / omega_a
    return Ae

def D2HPBW(F):
    # F is passed on in Hz, transferred to GHz in this function
    HPBW = 29.*240./(F/1e9) * np.pi / 180. / 60. / 60.
    return HPBW

def eta_mb_ruze(F, LFlimit, sigma):
    '''F in Hz, LFlimit is the eta_mb at => 0 Hz, sigma in m'''
    c = 299792458.
    eta_mb = LFlimit* np.exp(- (4.*np.pi* sigma * F/c)**2. )
    return eta_mb

def load_eta_atm():
    eta_atm_df = pd.read_csv(
        # os.path https://qiita.com/ymdymd/items/d758110d429f72bc10fb
        os.path.dirname(__file__)+'/data/atm.csv',
        skiprows=4,
        delim_whitespace=True,
        header=0
        )
    F_highres = eta_atm_df['F']
    return eta_atm_df, F_highres

def eta_atm_func(F, pwv, EL=60., eta_atm_df = pd.Series([]), F_highres = pd.Series([]), eta_atm_func_zenith = pd.Series([])):
    """
    Calculate eta_atm as a function of F by interpolation.
    If R~=0 then the function will average the atmospheric transmission
    within each spectrometer channel.
    There is a for loop aroudn the calculation of eta_atm, because otherwise the interp2d function in eta_atm_func_zenith orders the input by size.

    Parameters
    ----------
    F : scalar or vector
        Frequency of the astronomical signal.
        Unit: Hz (works also for GHz, will detect)
    pwv : float
        precipitable water vapour.
        Unit: mm
    EL : float
        telescope elevation angle
        Unit: degrees
    R : float
        spectral resolving power in F/W_F
            W_F is the 'equivalent bandwidth'
                http://www.astrosurf.com/buil/us/spe2/hresol7.htm
        R is used to average the atmospheric trannsmission within
        one spectrometer channel.
        If R = 0, then the function will return the transmission at that
        exact frequency.
        Unit : None.

    Returns
    -------
    eta_atm : float (if F is scalar) or a 1D np.array (if F is a vector)
        atmospheric tranmsmission.
        Units: None.
    """
    if np.average(F) > 10.**9:
        F = F / 10.**9
    if not hasattr(F, "__len__"):  # give F a length if it is an integer.
        F = np.asarray([F])
    if eta_atm_df.empty:
        eta_atm_df, F_highres = load_eta_atm()
    if type(eta_atm_func_zenith) != interp2d:
        eta_atm_func_zenith = eta_atm_interp(eta_atm_df)
    pwv = np.squeeze(pwv)
    eta_atm = np.zeros((len(F), len(pwv)))
    for i in range(len(pwv)):
        eta_atm = np.abs(eta_atm_func_zenith(pwv[i], F)) ** (1./np.sin(EL*np.pi/180.))
    return np.squeeze(eta_atm)


def eta_atm_interp(eta_atm_dataframe):
    """
    Used in the function eta_atm_func().
    Returns a function that interpolates atmospheric transmission data
    downloaded from ALMA
    (https://almascience.eso.org/about-alma/atmosphere-model)
    The returned function has the form of
    eta = func(F [GHz], pwv [mm]). Note telescope EL = 90 (zenith)

    Parameters
    ----------
    eta_atm_dataframe : pandas.DataFrame

    Returns
    --------
    func : function that returns the atmospheric transmission

    Example
    --------
        % read csv file with pandas (in e.g., Jupyter)
        eta_atm_df = pd.read_csv("<desim-folder>/data/atm.csv",skiprows=4,
                                 delim_whitespace=True,header=0)
        % make function from pandas file
        etafun = desim.eta_atm_interp(eta_atm_df)

    """
    x = np.array(list(eta_atm_dataframe)[1:]).astype(np.float)
    y = eta_atm_dataframe['F'].values
    z = eta_atm_dataframe.iloc[:, 1:].values
    func = interp2d(x, y, z, kind='cubic')
    return func


def rad_trans(rad_in, medium, eta):
    """
    Calculates radiation transfer through a semi-transparent medium.
    One can also use the same function for
    Johnson-Nyquist PSD (power spectral density) instead of temperature.

    Parameters
    ----------
    rad_in : scalar or vector
        brightness temperature (or PSD) of the input
        Units: K (or W/Hz)
    medium : scalar
        brightness temperature (or PSD) of the lossy medium
        Units: K (or W/Hz)
    eta : scalar or vector
        transmission of the lossy medium
        Units: K (or W/Hz)

    Returns
    -------
    rad_out : brightness temperature (or PSD) of the output

    """
    rad_out = eta * rad_in + (1 - eta) * medium
    return rad_out


def window_trans(
        F,
        psd_in,
        psd_cabin,
        psd_co,
        thickness=8.e-3,    # in m
        tandelta=4.805e-4,  # tan delta, measured Biorad
        tan2delta=1.e-8,
        neffHDPE=1.52,
        window_AR = True
        ):
    """
    Calculates the window transmission.

    Parameters
    ----------
    F : scalar or vector.
        Frequency
        Units: Hz
    psd_in : scalar or vector.
        PSD of the incoming signal.
        Units : W / Hz
    psd_cabin : scalar or vector.
        Johnson-Nyquist PSD of telescope cabin temperature.
        Units : W / Hz
    psd_co : scalar or vector.
        Johnson-Nyquist PSD of cold-optics temperature.
        Units : W / Hz
    thickness: scalar or vector.
        thickness of the HDPE window.
        Units: m
    tandelta, tan2delta : scalar
        values from Stephen.
            "# 2.893e-8 %% tan delta, measured Biorat. I use 1e-8 as this fits
            the tail of the data better"
    neffHDPE : scalar
        refractive index of HDPE. set to 1 to remove reflections.
        Units : None.

    Returns
    -------
    psd_after_2nd_refl : scalar or vector
        PSD looking into the window from the cold optics
    eta_window : scalar or vector
        transmission of the window

    """
    # Parameters to calcualte the window (HDPE), data from Stephen
    # reflection. ((1-neffHDPE)/(1+neffHDPE))^2. Set to 0 for Ar coated.

    if window_AR is True:
        HDPErefl = 0.
    else:
        HDPErefl = ((1-neffHDPE)/(1+neffHDPE))**2

    eta_HDPE = np.exp(-thickness * 2 * np.pi * neffHDPE *
                      (tandelta * F / c + tan2delta * (F / c)**2))
    if type(eta_HDPE) != np.float64:
        eta_HDPE = eta_HDPE.reshape([eta_HDPE.shape[0], 1])
    # most of the reflected power sees the cold.
    psd_after_1st_refl = rad_trans(psd_in, psd_co, 1.-HDPErefl)
    psd_before_2nd_refl = rad_trans(psd_after_1st_refl, psd_cabin, eta_HDPE)
    # the reflected power sees the cold.
    psd_after_2nd_refl = rad_trans(psd_before_2nd_refl, psd_co, 1.-HDPErefl)

    eta_window = (1.-HDPErefl)**2 * eta_HDPE

    return psd_after_2nd_refl, eta_window


def nph(F, T):
    """
    Photon occupation number of Bose-Einstein Statistics.
    If it is not single temperature, use nph = Pkid/(W_F * h * F)

    Parameters
    ----------
    F : scalar or vector.
        Frequency
        Units: Hz
    T : scalar or vector.
        temperature
        Units: K

    Returns
    --------
    n : scalar or vector
        photon occupation number
        Units : None.
    """
    n = 1./(np.exp(h*F/(k*T))-1.)
    return n


def johnson_nyquist_psd(F, T):
    """
    Johnson-Nyquist power spectral density.
    Don't forget to multiply with bandwidth to caculate the total power in W.

    Parameters
    ----------
    F : scalar or vector.
        Frequency
        Units: Hz
    T : scalar or vector.
        temperature
        Units: K

    Returns
    --------
    psd : scalar or vector
        Power Spectral Density.
        Units : W / Hz
    """
    psd = h*F*nph(F, T)
    return psd


def T_from_psd(
        F,
        psd,
        method='Planck'
        ):
    """
    Calculate Planck temperature from the PSD a single frequency,
    or an array of frequencies.

    Parameters
    ----------
    F : scalar or vector.
        Frequency
        Units: Hz
    psd : scalar or vector
        Power Spectral Density.
        Units : W / Hz
    method: optional, sring.
        default: 'Planck'
        option: 'Rayleigh-Jeans'
    Returns
    --------
    T : scalar or vector.
        Planck temperature.
        Units : K

    """
    if type(F) != np.float64:
        F = F.reshape([F.shape[0], 1])
    if method == 'Planck':
        T = h*F/(k*np.log(h*F/psd+1.))
    elif method == 'Rayleigh-Jeans':
        T = psd / k
    else:
        sys.exit("Error: Method should be Planck or Rayleigh-Jeans.")

    return T

def photon_NEP_kid(
        F,
        Pkid,
        W_F
        ):
    """
    NEP of the KID, with respect to the absorbed power.

    Parameters
    -----------
    F:  Frequency of the signal responsible for loading.
        Unit: Hz
    Pkid: Power absorbed by the KID.
        Unit: W
    W_F: detection bandwidth, with respect to the power that sets the loading.
        Unit: Hz

    Note
    --------
    Pkid/(W_F * h * F) gives the occupation number.
    """
    # photon_term = 2 * Pkid * (h*F + Pkid/W_F)
    poisson_term = 2 * Pkid * h * F
    bunching_term = 2 * Pkid * Pkid / W_F
    r_term = 4 * Delta_Al * Pkid / eta_pb
    NEPkid = np.sqrt(poisson_term + bunching_term + r_term)
    return NEPkid
"""
def MDLF_simple(
        pwv = 0.5, # Precipitable Water Vapor in mm
        EL = 60., # Elevation angle in degrees
        snr = 5., # Target S/N of the detection
        obs_hours = 8. # Total hours of observation, including ON-OFF and calibration overhead
        ):

    Fmin = 220
    Fmax = 440

    F = np.linspace(Fmin, Fmax, (Fmax - Fmin)*10 + 1)*1e9

    # Main beam efficiency of ASTE
    eta_mb = eta_mb_ruze(F=F,LFlimit=0.8,sigma=37e-6) * 0.9 # see specs, 0.9 is from EM, ruze is from ASTE

    D2goal_input ={
        'F' : F, # Frequency in GHz
        'pwv':pwv, # Precipitable Water Vapor in mm
        'EL':EL, # Elevation angle in degrees
        'theta_maj' : D2HPBW(F), # Half power beam width (major axis)
        'theta_min' : D2HPBW(F), # Half power beam width (minor axis)
        'eta_mb' : eta_mb, # Main beam efficiency
        'snr' : snr, # Target S/N of the detection
        'obs_hours' :obs_hours, # Total hours of observation, including ON-OFF and calibration overhead
        'on_source_fraction':0.4*0.9 # ON-OFF 40%, calibration overhead of 10%
    }

    D2goal = spectrometer_sensitivity(**D2goal_input)

    D2baseline_input = {
        'F' : F,
        'pwv':pwv,
        'EL':EL,
        'eta_circuit' : 0.32 * 0.5, # <= eta_inst Goal 16%, Baseline 8%
        'eta_IBF' : 0.4, # <= Goal 0.6
        'KID_excess_noise_factor' : 1.2, # Goal 1.1
        'theta_maj' : D2HPBW(F), # Half power beam width (major axis)
        'theta_min' : D2HPBW(F), # Half power beam width (minor axis)
        'eta_mb' : eta_mb,
        'snr' : snr,
        'obs_hours' :obs_hours,
        'on_source_fraction':0.3*0.8 # <= Goal 0.4*0.9
    }

    D2baseline = spectrometer_sensitivity(**D2baseline_input)

    ### Plot-------------------

    fig, ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(D2baseline['F']/1e9,D2baseline['MDLF'],'--',linewidth=1,color='b',alpha=1,label='Baseline')
    ax.plot(D2goal['F']/1e9,D2goal['MDLF'],linewidth=1,color='b',alpha=1,label='Goal')
    ax.fill_between(D2baseline['F']/1e9,D2baseline['MDLF'],D2goal['MDLF'],color='b',alpha=0.2)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Minimum Detectable Line Flux ($\mathrm{W\ m^{-2}}$)")
    ax.set_yscale('log')
    ax.set_xlim(Fmin-20,Fmax+20)
    ax.set_ylim([10**-20,10**-17])
    ax.tick_params(direction='in',which='both')
    ax.grid(True)
    ax.set_title("$R="+str(int(D2goal['R'][0]))+", snr=" + str(D2goal['snr'][0]) + ',\ t_\mathrm{obs}='
                 +str(D2goal['obs_hours'][0])
                 +'\mathrm{h}$ (incl. overhead), PWV=' + str(D2goal['PWV'][0]) + "mm, EL="+str(int(D2goal['EL'][0]))+'deg',
                 fontsize=12)
    ax.legend()
    plt.tight_layout()

    # Create download link
    #................................

    df_download = D2goal[['F','MDLF']]
    df_download = df_download.rename(columns={'MDLF':'MDLF (goal)'})
    df_download = df_download.join(D2baseline[['MDLF']])
    df_download = df_download.rename(columns={'MDLF':'MDLF (baseline)'})

    return create_download_link(df_download,filename='MDLF.csv')
"""
def D2HPBW(F):
    HPBW = 29.*240./(F/1e9) * np.pi / 180. / 60. / 60.
    return HPBW

def eta_mb_ruze(F, LFlimit, sigma):
    '''F in Hz, LFlimit is the eta_mb at => 0 Hz, sigma in m'''
    c = 299792458.
    eta_mb = LFlimit* np.exp(- (4.*np.pi* sigma * F/c)**2. )
    return eta_mb

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index =False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

def lineflux(
    Lfir = 5.e+13, ## total infrared luminosity (L_Sun)
    switch_dwarf = False ## would you like to use line-to-TIR ratios for dwarf galaxies?
    ):
    ### Calculation of Astronomical line, developed by Y. Tamura --------

    ## line-to-TIR luminosity ratio (L_Sun or Watt)
    Rcii_B08, Roiii_B08, Roi_B08 = 1.3e-3, 8.0e-4, 1.0e-3 ## from Brauer+2008
    Rcii_DGS, Roiii_DGS, Roi_DGS = 2.5e-3, 5.0e-3, 1.7e-3 ## from Cormier+2015

    ## rest frequency (GHz)
    f_cii, f_oiii, f_oi = 1900.5369, 3393.00062, 4744.8

    z_Dl_df = pd.read_csv(
        # os.path https://qiita.com/ymdymd/items/d758110d429f72bc10fb
        os.path.dirname(__file__)+'/data/z_Dl.csv',
        skiprows=0,
        delim_whitespace=False,
        header=0
        )

    z = z_Dl_df.values[:,0]
    Dl = z_Dl_df.values[:,1]

    Dl_at_z = interp1d(z, Dl)

    ## redshift
    #     exec(open(desim_path + '/Dl_at_z2.py').read())
    #z = concatenate( (arange(0.1,0.5,0.1), arange(0.5,2,0.5), arange(2,6,1)), 1)
    #z = concatenate( (arange(0.1,0.5,0.1), arange(0.5,2,0.5), arange(2,12,1)), 1)
    #z = concatenate( (arange(0.1,1,0.1), arange(1,8,0.2)), 1)
    z = np.array([3,4,5,6,7,8,9,10,11,12])
    d_l = Dl_at_z(z) ## luminosity distance (Mpc)

    Fcii_B08  = flux_from_line_luminosity(z, d_l, f_cii/(1+z),  Lfir * Rcii_B08)
    Foiii_B08 = flux_from_line_luminosity(z, d_l, f_oiii/(1+z), Lfir * Roiii_B08)
    Foi_B08   = flux_from_line_luminosity(z, d_l, f_oi/(1+z),   Lfir * Roi_B08)

    Fcii_DGS  = flux_from_line_luminosity(z, d_l, f_cii/(1+z),  Lfir * Rcii_DGS)
    Foiii_DGS = flux_from_line_luminosity(z, d_l, f_oiii/(1+z), Lfir * Roiii_DGS)
    Foi_DGS   = flux_from_line_luminosity(z, d_l, f_oi/(1+z),   Lfir * Roi_DGS)

    if switch_dwarf:
        Fcii_DGS, Foiii_DGS, Foi_DGS = Fcii_DGS, Foiii_DGS, Foi_DGS
    else:
        Fcii_DGS, Foiii_DGS, Foi_DGS = Fcii_B08, Foiii_B08, Foi_B08

    return Fcii_DGS, Foiii_DGS, Foi_DGS, f_cii, f_oiii, f_oi, z

def co_luminosity(z, d_l, f_obs, int):

	c1, c2 = 3.25e+07, 1.04e-03

	#Lp  = c1 * int * d_l**2 / (f_obs**2 * (1 + z)**3 )
	L   = c2 * int * d_l**2 * f_obs

	return L


def flux_from_line_luminosity(z, d_l, f_obs, L):
	"""
		d_l - luminosity distance (Mpc)
		f_obs - observing frequency (GHz)
		L - line luminosity (L_Sun)
	"""

	L_for1Jykms = co_luminosity(z, d_l, f_obs,  1.000) ## Lsun
	F_for1Jykms = 1.000 * 1e-26 * (f_obs  * 1e9 / 299792.458) ## W m^-2

	return L * (F_for1Jykms / L_for1Jykms)
