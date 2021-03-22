[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4279085.svg)](https://doi.org/10.5281/zenodo.4279085)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square)](LICENSE)

Time-dependent end-to-end model for post-process optimization of the DESHIMA spectrometer.

## TL;DR
TiEMPO allows for simulating the output of the DESHIMA spectrometer. The model features simulation of an input galaxy with atomic spectral lines that is sampled with the ABBA chopnod method. The simulation includes atmospheric distrotion through distortions in the optical thickness of the atmosphere due to precipitable water vapor using outputs of the ARIS model (provided by the user), telescope transmission and finally noise and attenuation due to the MKID filterbank of DESHIMA.

## Output of the model 
The model outputs the following data: 
1. Vector of the time: all moments in time at which the signal is calculated in s.
2. Matrix of the power: matrix of all the power values of the signal in W. It has dimensions [5, #filters, #timesamples]. The first dimension equals 5, as 5 pwv values are taken in each timesample. 
3. Matrix of the sky temperature: matrix of the power converted to sky temperature (has the same dimensions as the matrix of the power) in K. 
4. Center frequencies of the filters: The center frequencies of the filters in the filterbank in the MKID chip in Hz. 

The pwv values are taken in the following order: 

![pwv values](https://raw.githubusercontent.com/deshima-dev/tiempo_deshima/master/skychopping.png)
1. Left position
2. Center position with galaxy
3. Right position
4. Top position
5. Bottom position
6. Center position without galaxy
The pwv values in position 2 and position 4 are equal, but otherwise the sky temperatures are computed separately.

## Using the model
### Example
```
time_vector, center_freq = tiempo_deshima.run_tiempo(input_dictionary = 'deshima_2', prefix_atm_data = 'aris200602.dat-', sourcefolder = '../Data/output_ARIS', save_name_data = 'TiEMPO_simulation')
```
Inputs include:

### Atmosphere
**pwv_0** (*float*): The value of the precipitable water vapor that is added to the dpwv from ARIS in mm. 
**windspeed** (*float*): The windspeed of the atmosphere in m/s.
**prefix_atm_data** (*string*): The beginning of the name with which the atmosphere data is saved. For example, if the files are called *sample-00.dat-000*, *sample-00.dat-001* etc, then Prefix_atm_data must be 'sample-00.dat-'
**sourcefolder** (*string*): folder in which the atmosphere data is saved (relative to cwd)
**grid** (*float*): The width of a grid square in the atmosphere map in m
**max_num_strips** (*integer*): The number of atmosphere strips that are saved as ARIS output.
**x_length_strip** (*int*): The length of one atmosphere strip in the x direction. This is the number of gridpoints, *not* the distance in meters.  
**separation** (*float*): Separation between two chop positions in m, assuming that the atmosphere is at 1km height. Default is 1.1326 (this corresponds to 116.8 arcsec).
**useDESIM** (*bool*): Determines whether the simple atmospheric model is used (0) or the more sophisticated desim simulation (1).
**inclAtmosphere** (*bool*):Determines whether the simple atmospheric model is used (0) or the more sophisticated desim simulation (1).

### Galaxy
**luminosity** (*float*): Luminosity of the galaxy, in Log(L_fir [L_sol])
**redshift** (*float*): The redshift of the galaxy
**linewidth** (*float*): The linewidth, in km/s
**num_bins** (*int*): Determines the amount of bins used in the simulation of the galaxy spectrum. 
**galaxy_on** (*bool*): Can be used to turn the galaxy in position 2 off. Default is True (galaxy is present).

### Observation
**EL** (*float*): The elevation of the telescope, in degrees
**EL_vec** (*vector of floats*): If this parameter is set, it allows to specify the elevation of the telescope in degrees per timestep, for example in the case of tracking a target. Vector must have a length of 160Hz times obs_time.
**obs_time** (*float*): The observation time. This parameter has to be smaller than **max_obs_time**, which is calculated using the windspeed and the total length of the strips of atmosphere data, in s.

### Instrument
**F_min** (*float*): Lowest center frequency of all the MKIDs.
**spec_res** (*float*): Spectral resolution
**f_spacing** (*float*): spacing between center frequencies = F/dF (mean).
**num_filters** (*float*): Number of filters in the filterbank
**beam_radius** (*float*): Radius of the Gaussian telescope beam in meters.

### Miscellaneous
**input_dictionary** (*string*): Determines where the input values of keywords F_min thru come from: either standard values for DESHIMA, manual entry from the keywords or from a txt file 
**dictionary_name** (*string*): name of a txt file in which the values of optional keywords are saved.
**save_name_data** (*string*): The name with which the produced data is saved.
**savefolder** (*string*): Folder in which the produced data is saved (relative to cwd)
**save_P** (*bool*): determines whether power in Watts is saved
**save_T** (*bool*): determines whether sky temperature in Kelvins is saved
**n_jobs** (*int*): amount of threads in the threadpool
**n_batches** (*int*): amount of batches in which the output data is divided into in time

## Important instructions

### Atmosphere
* All atmosphere strips must have the same length in the x direction and a length in the y direction of at least 30 gridpoints. ('length' means number of gridpoints, *not* distance in meters)

### Changing the number of filters or the distribution of the center frequencies of the filters
* For each filter, an interpolation between the power and the sky temperature is made. This means that these interpolations need to be made and saved again if the center frequencies of the filters are changed, before TiEMPO can be run again. This can be done by using ```new_filterbank()``` with the desired input dictionary, which can be generated using ```get_dictionary()```.
* Since the chip properties are altered, 'deshima_1' and 'deshima_2' cannot be used as keywords for *input_dictionary* anymore.

#### Example of changing the filters
```
dict = tiempo_deshima.get_dictionary(input_dictionary = 'manual', prefix_atm_data = 'aris.dat-', sourcefolder = '../Data/output_ARIS', save_name_data = 'TiEMPO_simulation_new_filters')
tiempo_deshima.new_filterbank(dict)
time_vector, center_freq = tiempo_deshima.run_tiempo(input_dictionary = 'manual', prefix_atm_data = 'aris.dat-', sourcefolder = '../Data/output_ARIS', save_name_data = 'TiEMPO_simulation_new_filters')
```
## Installation
```
pip install tiempo_deshima
```