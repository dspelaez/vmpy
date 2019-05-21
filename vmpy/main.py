#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# vim:fdm=marker
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

from __future__ import unicode_literals

"""
File:     vmpy.py
Created:  2017-07-18 11:25
"""

# --- import libs ---
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy.signal as signal
import scipy.stats as stats
import os, sys, glob, pickle
#
try:
    import gsw
except:
    raise ImportError("You must install gsw library `>> pip install gsw`")
#
from scipy.io import loadmat
#
from tools.helpers import scientific_notation



# --- butterworth filter --- {{{
def butterworth(x, fs, smooth=0.5, order=2, kind="low"):
    """Implement the n-order butterworth filter and return the filtered signal.
    
    Args:
        x (array): time or space signal
        f (array): sampling frequency
        smooth (float): smoothing factor
        order (float): order of the filter
        kind (str): kind of filter `low` or `high`
    
    Returns:
        array: filtered signal
    """
    
    # chech arguments
    if kind not in ['low', 'high']:
        raise Exception("kind must be low or high")

    # padding length
    padlen = int(min(len(x), 2 * np.floor(fs/smooth)))

    # filter coefficients
    f_nyq = fs / 2.
    b, a = signal.butter(order, smooth / f_nyq, kind)

    # perform filter and return filtered signal
    return signal.filtfilt(b, a, x, padtype='even', padlen=padlen)
# --- }}}

# --- moving average filter --- {{{
def movavg(x, ns, window='hann'):
    """Low pass filter using a Hann windonw.

    This function performs a moving average filter using a tapering window,
    resulting in a low-pass filter or smoothing
    
    Args:
        x (array): time or space signal
        ns (int): number of points to average
        window (str): see signal.get_window

    Returns:
        array: filtered signal
    """

    # get window and normalize
    win  = signal.get_window(window, ns)
    win /= np.sum(win)

    return signal.convolve(x, win, 'same')
# --- }}}

# --- despike --- {{{
def despike(x, fs=512, smooth=0.5, tpad=0.04, threshold=8,
            n_passes=1, plot=False, filename=None):

    """Remove spikes of a given signal.
    
    This function apply the despike method consisting in comparing the
    signal with a low passed signal and detect the values larger than a
    threshold. This procedure is performed iteratively until no spikes
    are encountered.

    Args:
        x (array): signal to be despiked.
        fs (float): sampling frequency.
        smooth (float): smoothing factor.
        tpad (float): time interval around at each side of the spike to be
            marked as invalid.
        threshold (float): value to detect the anomalies.
        n_passes (integer): times to be applied the method.
        plot (bool): choose if you wanna plot some nice figs, the figures are
            important to do a visual inspection.
        filename (str): if filename is not None, the figure is saved as
            specified.

    Returns:
        array: clean signal
    """
    
    # --- filter signal ----
    
    # high pass the signal and rectify
    x_HP = abs(butterworth(x, fs=fs, smooth=0.5, order=1, kind="high"))

    # smoothed signal
    x_LP = butterworth(x_HP, fs=fs, smooth=smooth, order=1, kind="low")

    # find spikes
    bool_spikes = x_HP / x_LP > threshold
    indx_spikes = bool_spikes.nonzero()[0]

    # return signal if no spikes where found
    n_spikes = len(indx_spikes)
    if n_spikes == 0:
        return x


    # --- find and replace spikes ---

    # create copies of x
    x_dirty = x.copy()
    x_clean = x.copy()
    x_masked = x.copy()

    # determinte lenght to pad and to get the mean
    npad = int(tpad * fs / 2)
    nextend = int(fs / (4.* smooth))

    # for each spike
    for s in indx_spikes:
        #
        # find adjacent points to remove
        i_beg = max(0, s-npad)
        i_end = min(len(x)-1, s+npad)
        #
        # mark 2*npad points as invalid
        bool_spikes[i_beg:i_end] = True
        x_masked[i_beg:i_end] = np.nan
        #
        # find adjacent points to average
        j_beg = max(0, s-nextend)
        j_end = min(len(x)-1, s+nextend)
        mean = np.nanmean(x_masked[j_beg:j_end])
        #
        # replace invalid values by mean of 2*nextend points
        if np.isnan(mean):
            mean = 0.
        x_clean[i_beg:i_end] = mean


    # number of removed points
    n_removed = len(np.isnan(x_masked).nonzero()[0])


    # funcion to make plot
    def make_figure():
        #
        title = f"Pases: {n_passes}, "\
                f"Datos anómalos: {n_spikes}, "\
                f"Eliminados: {n_removed}"
        time = np.linspace(0, len(x) / fs, len(x))
        xmax = np.ceil(max(abs(x)))
        #
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        #
        ax1.plot(time, x_HP, '0.75', label="Señal rectificada")
        ax1.plot(time, x_LP, 'k', lw=1.5, label="Pseudo desviación")
        ax1.plot(time[bool_spikes], x_HP[bool_spikes], '.b', 
                ms=2, label="Datos eliminados")
        ax1.plot(time[indx_spikes], x_HP[indx_spikes], '*r', 
                ms=3, label=u"Datos anómalos")
        ax1.legend(loc=0, ncol=2, facecolor="w")
        ax1.set_ylim((0., xmax-1))
        #
        ax2.plot(time, x_dirty, lw=1.5, label="Observada")
        ax2.plot(time, x_clean, label="Depurada")
        ax2.legend(loc=0, ncol=2, facecolor="w")
        ax2.set_xlabel("Tiempo [s]")
        ax2.set_ylim((-xmax, xmax))
        #
        ax1.set_title(title)
        #
        if filename is not None:
            fig.savefig(f"{filename}_pass{n_passes}.png", dpi=600)


    # --- apply despike n times ---
    if n_passes >= 10:
        return x_clean
    else:
        if plot:
            make_figure()
        return despike(x_clean, fs=fs, tpad=tpad, threshold=threshold,
                       n_passes=n_passes+1, plot=plot, filename=filename)
# --- }}}

# --- viscosity --- {{{
def viscosity(P, T, S, rho=None):
    """
    Computes kinematic viscosity of sea water.

    Args:
        P (float or array): Pressure [dBar]
        T (float or array): Temperature [degC]
        rho (float or array): In-situ density [kg/m^3]

    Returns:
        float or array: Kinematic viscosity [m^2/s]
    
    References:
        * Millero, J. F., 1974, The Sea, Vol 5, M. N. Hill,
          Ed, John Wiley, NY, p. 3.
        * Peters and Siedler, in Landolt-Bornstein New Series V/3a (Oceanography),
          pp.234
    """

    # check if density has passed as an argument
    if rho is not None:
        pass
    else:
        rho = density(P, T, S)

    # define coefficients according to RSI
    a0, a1 = 3.5116e-6, 1.2199e-6
    b0, b1 = 1.4342e-6, 1.8267e-8
    c0 = 1.002e-3
    d0, d1, d2 = 1.1709, 1.827e-3, 89.93

    # molecular viscosity for fresh water
    mu0 = c0 * 10. ** ((d0 * (20 - T) - d1 * (T - 20)**2) / (T + d2))

    # molecular viscosity for a given salinity
    mu = mu0 * (1 + (a0 + a1*T) * (rho*S)**0.5 + (b0 + b1*T) * rho*S)

    return mu / rho
# --- }}}

# --- seawater --- {{{
def seawater(P, T, C, clean=True, **kwargs):
    """
    This function uses the Gibbs sea water routines in order to compute the
    salinity as a function of the conductivity, tepmperature and pressure. Also
    this function interpolates the conductivity and temperature if their lengh
    does not coincide with the pressure array.

    Arguments:
        P: Pressure [dBar]
        T: Temperature [degC]
        C: Conductivity [mS/cm]
        clean: If True perform a despiking before output 

    Returns:
        SA: Absolute salinity
    """

    # parse optional arguments
    lon = kwargs.get("lon", -116.83)
    lat = kwargs.get("lat",   31.81)

    # ckeck if len 1
    if isinstance(P, float) or isinstance(P, int):
        SP = gsw.SP_from_C(C, T, P)
        SA = gsw.SA_from_SP(SP, P, lon, lat)
        return SA
    
    # despike temperature and conductivity signals
    threshold = kwargs.get('threshold', 8)
    if clean:
        T = despike(T, threshold=threshold)
        C = despike(C, threshold=threshold, smooth=1.0)

    # resample conductivity
    if len(C) != len(P):
        x_old = np.linspace(0, 1, len(C))
        x_new = np.linspace(0, 1, len(P))
        C_new = np.interp(x_new, x_old, C)
    else:
        C_new = C.copy()

    # resample temperature
    if len(T) != len(P):
        x_old = np.linspace(0, 1, len(T))
        x_new = np.linspace(0, 1, len(P))
        T_new = np.interp(x_new, x_old, T)
    else:
        T_new = T.copy()

    # compute practical salinty, absolute salinity, conservative
    # temperature, in-situ density, and viscosity
    SP = gsw.SP_from_C(C_new, T_new, P)
    SA = gsw.SA_from_SP(SP, P, lon, lat)
    CT = gsw.CT_from_t(SA, T_new, P)
    rho = gsw.rho(SA, CT, P)
    kvisc = viscosity(P, T_new, SA, rho)

    # return variables
    return SA, rho, kvisc

# --- }}}

# --- shear velocity spectrum --- {{{
def shear_spectrum(t, a=None, b=None, fft_params={}, **kwargs):
    """
    This function computes the shear spectrum and correct it from vibrations
    along axis of sensititvity computing the coherence and substracts it from
    the shear spectrum

    Arguments:
        t: Dictionary containing time series of the profile. The variables used are:
            * sh1_clean, sh2_clean --> shear despiked and filtered 
            * Ax, Ay --> acceleration in both directions
    
        a, b: Indices of the limits in the time series [def: None]

        fft_params: Dictionary containing: [See signal.welch]
            * fs       -> sampling frquency
            * nperseg  -> lenght of each segment
            * noverlap -> points to overlap between segments
            * window   -> string describing the tapering window or array

    Optional arguments:
        return_all: If return_all = True, then all computer frequency spectra
            are returned in a dictionary. If return_all = False only frequency (or
            wavenumber) and P_sh1 and P_sh2 are returned. Finally, if return_all ==
            None nothing is returned, instead a figure is plotted.

        poly_deg: The detrending procedure consists in fittin a polynomial curve
            and substract it from time series. This variables is the power of the
            polynomial. Default is 2 which means parabolic detrending.

    References:
        * https://rocklandscientific.com/support/knowledge-base/technical-notes/
                 
    """

    # --- get the variales ---
    sh1, sh2 = t["sh1_clean"][a:b], t["sh2_clean"][a:b]
    Ax, Ay   = t["Ax"][a:b], t["Ay"][a:b]
    P, W     = t["P_fast"][a:b], t["W_fast"][a:b]
    

    # --- get fourier transform params ---
    fs       = fft_params.get('fs', 512)
    nperseg  = fft_params.get('nperseg', 1024)
    noverlap = fft_params.get('noverlap', 512)
    window   = fft_params.get('window', 'hann')


    # --- detrend using parabolic fit ---
    N = len(P)
    poly_deg = kwargs.get("poly_deg", 2)
    if poly_deg == 0:
        sh1_detrended = sh1.copy()
        sh2_detrended = sh2.copy()
        Ax_detrended  = Ax.copy()
        Ay_detrended  = Ay.copy()
    else:
        sh1_detrended = sh1 - np.polyval(np.polyfit(P, sh1, poly_deg), P)
        sh2_detrended = sh2 - np.polyval(np.polyfit(P, sh2, poly_deg), P)
        Ax_detrended  = Ax  - np.polyval(np.polyfit(P, Ax,  poly_deg), P)
        Ay_detrended  = Ay  - np.polyval(np.polyfit(P, Ay,  poly_deg), P)


    # --- create arrays with detrend time series ---
    U = np.vstack((sh1_detrended, sh2_detrended))
    A = np.vstack((Ax_detrended,  Ay_detrended))

    
    # ---- compute cross-power spectral density matrix using scipy.signal ---

    # define function for speed
    csd = lambda x, y: signal.csd(x, y, **fft_params)[1] 

    # compute frequency array
    f  = signal.csd(U, A, **fft_params)[0]

    
    # --- perform correction ---
    correction_status = "uncorrected"
    try:
        # shear cross and auto-spectra
        UU = np.array([[csd(U[i], U[j]) for j in range(2)] for i in range(2)])
        UA = np.array([[csd(U[i], A[j]) for j in range(2)] for i in range(2)])
        AA = np.array([[csd(A[i], A[j]) for j in range(2)] for i in range(2)])
        
        # compute correction for each component
        clean_UU = np.zeros((2,2,len(f)), dtype='complex')
        for i in range(len(f)):
            clean_UU[:,:,i] = UU[:,:,i] -\
            UA[:,:,i].dot(np.linalg.inv(AA[:,:,i])).dot(np.conj(UA[:,:,i]).T)

        # save status variables
        correction_status = "matrix"
    
    except:
        # if matricial correction fails permform element-wise correction
        clean_UU = UU - UA * np.conj(UA.T) / AA
        correction_status = "elementwise"
    

    # --- extract shear spectrum corrected and convert it to wavenumber ---

    # mean pressure and falls speed
    Pmean = np.nanmean(P)
    Wmean = np.nanmean(W)

    # wavenumber from taylor frozen field hypothesis
    k = f / Wmean

    # correction factor due to the probe response function
    correction = 1. + (k / 48)**2
    correction[k>150] = 1. + (150. / 48.)**2

    # spectra of acceleration
    P_Ax = AA[0,0,:].real * Wmean
    P_Ay = AA[1,1,:].real * Wmean

    # spectra of shear
    P_sh1 = UU[0,0,:].real * Wmean * correction
    P_sh2 = UU[1,1,:].real * Wmean * correction
    P_sh1_clean = clean_UU[0,0,:].real * Wmean * correction
    P_sh2_clean = clean_UU[1,1,:].real * Wmean * correction


    # --- degrees of freedom and significance test ---
    dof = int(2 * np.sqrt(8/3) * (2 * N  / nperseg - 1))
    low_limit = dof / stats.chi2.ppf(0.975, dof)
    upp_limit = dof / stats.chi2.ppf(0.025, dof)


    # --- return variables ---
    return_all = kwargs.get('return_all', False)
    if return_all:
        out                      = {}
        out["f"]                 = f
        out["k"]                 = k
        out["Wmean"]             = Wmean
        out["Pmean"]             = Pmean
        out["dof"]               = dof
        out["P_Ax"]              = P_Ax
        out["P_Ay"]              = P_Ay
        out["P_sh1"]             = P_sh1
        out["P_sh2"]             = P_sh2
        out["P_sh1_clean"]       = P_sh1_clean
        out["P_sh2_clean"]       = P_sh2_clean
        out["correction_status"] = correction_status
        return out
    else:
        return k, P_sh1_clean, P_sh2_clean

# --- }}}

# --- nasmyth spectrum --- {{{
def nasmyth(epsilon=1.1464E-6, nu=1.1464E-6, k=1000):
    """
    This funcion returns the Nasmyth spectrom for a given rate of dissipation of
    TKE, kinematic viscosity and wavenumber array

    Arguments:
        epsilon (float): rate of dissipation of TKE in [W/kg]
        nu (float): kinematic viscosity [default: 1E-6 m2/s] 
        k (int of array): wavenumber array or integer with number of
            wavenumbers.

    Returns:
        phi (array): Nasmyth spectrum
    """

    # Kolgomorov wavenumer
    ks = (epsilon/nu ** 3)**(1./4.)

    # create wavenumber array if an integer was given
    if isinstance(k, int):
        k = np.linspace(0, ks, k)

    # non dimensional wavenumerb x = k / ks
    x = k / ks

    # use equation modified by Lueck
    phi_adim = 8.05 * x**(1/3) / (1. + (20.6 * x) ** (3.715))
    return epsilon**(3/4) * nu**(-1/4) * phi_adim;

# --- }}}

# --- get dissipation rate --- {{{
def get_dissrate(k, P_sh, Wmean, nu=1.464E-6):
    """
    This function applies the RSI algorith to obtain the rate of disipation of
    TKE from the wavenumber shear spectrum P_sh. This algorithm is presented in
    the Technical Note TN28 as part of the official documentation of the VMP250.

    This routine suposses that:
    * The signal was high-pass filtered between 0.5 and 2 cmp
    * The spikes was removed
    * The shear spectrum was corrected removing the coherent noise

    Arguments:
        k (array): wavenumber in cpm
        P_sh (array): clean wavenumber spectrum of vertical shear
        Wmean (float): mean fall speed
        nu (float): mean kinematic viscosity

    Returns:
        epsilon (float): rate of dissipation of TKE in W/kg

    References:
        * https://rocklandscientific.com/support/knowledge-base/technical-notes/

    """

    # --- common parameters ---
    fac = 7.5 * nu          # factor in the inegral
    a_parameter = 1.0774E9  # param to do first estimate
    e_high = 1.5E-5         # limit to chose the algorithm
    f_AA = 88.2             # antialiasing filter frequency
    k_AA = f_AA / Wmean     # antialiasing filter wavenumber
    x_95 = 0.1205           # nondimensional wavenumber for 95% variance
    x_isr = 0.02            # nondimensional wavenumber for inertial subrange 


    # --- define two useful functions ----
    # --- fit in range --- {{{
    def fit_in_range(value, interval):
        """
        This function fits the value into the interval
        """
        #
        if value < interval[0]:
            return interval[1]
        #
        elif value > interval[1]:
            return interval[1]
        #
        else:
            return value
    # --- }}}

    # --- inertial subrange fitting --- {{{
    def inertialsubrange_fitting(e):
        """
        This function performs a fitting to the +1/3 range for values of epsilon
        greater than a threshold, that is generally 3E-6
        """

        # --- find the fitting range ---
        k_max = np.min([x_isr * (e / nu**3)**.25, k_AA, 150])
        fit_range = k <= k_max
        index_limit = np.count_nonzero(fit_range) 

        # loop to ensure convergence
        for count in range(3):
            Psi_N = nasmyth(epsilon=e, nu=nu, k=k[fit_range])
            fit_error = np.log10(P_sh[1:index_limit] / Psi_N[1:index_limit]).mean()
            e = e * 10**(3*fit_error / 2)

        # --- remove up to 20 percent of the flyers ---

        # compute fit error
        Psi_N = nasmyth(epsilon=e, nu=nu, k=k[fit_range])
        fit_error = np.log10(P_sh[1:index_limit] / Psi_N[1:index_limit])
        flyers_bool = fit_error > 0.5
        if np.count_nonzero(flyers_bool) != 0:
            #
            # compute indices sortered descendently
            sorted_index = np.argsort(fit_error[flyers_bool])[::-1]
            #
            # define limit up to 20 percent of the data
            bad_limit = np.int(0.2 * len(fit_error)) + 1
            #
            # redifine index limit
            if len(sorted_index) > bad_limit:
                sorted_index = sorted_index[:bad_limit]
            #
            flyers_index = flyers_bool.nonzero()[0][sorted_index]
            fit_range[flyers_index] = False

        # redifine kmax
        k_max = k[fit_range][-1]

        # refit to the inertial subrange
        for count in range(3):
            Psi_N = nasmyth(epsilon=e, nu=nu, k=k[fit_range])
            fit_error = np.log10(P_sh[fit_range][1:] / Psi_N[1:]).mean()
            e = e * 10**(3*fit_error / 2)

        return e, k_max
    # --- }}}


    # --- initial estimation ----
    e10 = fac * np.trapz(P_sh[k<=10], x=k[k<=10]) # integrate until k = 10 cpm
    e1 = e10 * np.sqrt(1 + a_parameter * e10)     # first estimate using eq8 in TN28

    
    # --- variance method ---
    if e1 < e_high:

        # refine the dissipation estimate by fittin to the inertial subrange
        if  np.count_nonzero(k * (nu**3 / e1)**0.25 <= 0.02) >= 20:

            # apply isr_fitting function
            e2, _ = inertialsubrange_fitting(e1)
        #
        # use the estimate that we already have
        else:
            e2 = e1
        
        # compute a valid the limit of the integration
        with np.errstate(over="ignore"):
            k_95 = x_95 * (e2 / nu**3)**0.25
            valid_shear = k <= fit_in_range(np.min([k_AA, k_95]), [0, 150])
            index_limit = np.count_nonzero(valid_shear) 
        
        # compute the variables x and y to perform the polynomial fitting taking
        # into account that the first element has to be excluded
        x = np.log10(   k[1:index_limit])
        y = np.log10(P_sh[1:index_limit])

        # keep the fit oreder between 3 and 8
        fit_order = 3

        # check if we have enough points for polyfit
        if index_limit > fit_order + 2:
            # p   - polynomial of a polynomial fit to spectrum
            # pd1 - first derivative of the polynomial
            # pr1 - roots of first derivative of the polynomial
            p   = np.polyfit(x, y, fit_order)
            pd1 = np.polyder(p)
            pr1 = np.sort(np.roots(pd1))
            pr1 = np.array([i.real for i in pr1 if i.imag == 0])

            # filter roots so that only minima above 10 cpm remain
            if len(pr1) != 0:
                pr1 = pr1[np.polyval(np.polyder(pd1), pr1) > 0] # minima only
                pr1 = pr1[pr1 >= np.log10(10)] # spectral min must be at 10cpm or higher
                
                # fit root within a given range.
                if len(pr1) == 0:
                    pr1 = np.log10(k_95)
                else:
                    pr1 = pr1[0]
            else:
                pr1 = np.log10(k_95)
        
        # integrate the spectrum up to k_limit
        with np.errstate(over="ignore"):
            k_limit = fit_in_range(np.min([10**pr1, k_95, k_AA]), [0, 150])
            k_limit = k[k<=k_limit][-1]
            e3 = fac * np.trapz(P_sh[k<=k_limit], x=k[k<=k_limit])

        # next, the non-dimensional limit of integration
        x_limit = k_limit * (nu**3/e3)**0.25
        x_limit = x_limit**(4./3.)

        # next, the variance resolved according to Lueck's model
        var = np.tanh(48 * x_limit) - 2.9*x_limit*np.exp(-22.3 * x_limit)

        e_new = e3 / var
        done = 0
        while done == 0:
            x_limit = k_limit * (nu**3/e_new)**0.25
            x_limit = x_limit**(4./3.)
            var = np.tanh(48 * x_limit) - 2.9*x_limit*np.exp(-22.3 * x_limit)
            e_old = e_new
            e_new = e3 / var
            if e_new / e_old < 1.02:
                done = 1
                e3 = e_new

        # correct for missing variance at bottom depth of the spectrum
        phi_lower = nasmyth(epsilon=e3, nu=nu, k=k[1])
        e4 = e3 + 0.25 * fac * k[1] * phi_lower
        
        # re-loop to find dissipation rate
        if e4 /e3 > 1.1:
            e_new = e4 / var
            done = 0
            while done == 0:
                x_limit = k_limit * (nu**3/e_new)**0.25
                x_limit = x_limit**(4./3.)
                var = np.tanh(48 * x_limit) - 2.9*x_limit*np.exp(-22.3 * x_limit)
                e_old = e_new
                e_new = e4 / var
                if e_new / e_old < 1.02:
                    done = 1
                    e4 = e_new
        
        # store varibales
        epsilon = e4


    # --- inertial subrange fitting ---
    else:
        epsilon, k_limit = inertialsubrange_fitting(e1)


    # --- return results ---
    return epsilon, k_limit
# --- }}}



# --- main class ---- {{{
class VmpData(object):

    """Contains common routines to handle VMP-250 data.

    Main data structures
    --------------------

    t:
        This is a dictionaty containing all the info about the time series
        of the variables measured by the VMP and others derivated from those.

    p:
        This is a dictionay containing the binned variables computed from the
        original measured time series. It includes the "s" dictionary for each
        segment of the profile.

    s:
        This dictionary contains all the information of the spectrum of profile
        segment, such as, shear spectrum, acceleration spectrum, corrected
        spectrum, degrees of freedom, among others.
    """

    # --- default parameters --- {{{

    # directories
    cachepath = "/Users/dpelaez/Documents/Maestria/Cicese/tesis/data/vmp/cache"
    os.system(f"mkdir -p {cachepath}")

    # cleaning data
    shear_smooth = 0.5              # <--- high pass filter
    P_offset     = 0.0              # <--- pressure offset post-processing
    
    # spectral-related parameters
    fft_params = {}
    fft_params['fs']       = 512    # <--- default sampling rate
    fft_params['nperseg']  = 1024   # <--- points per segment
    fft_params['noverlap'] = 512    # <--- points to overlap
    fft_params['window']   = 'hann' # <--- tappering window
    
    # binned profiles parameters
    P_min = 3                       # <--- first value of each profile
    window_length = 2**12           # <--- this corresponds roughly to 8 seconds
    overlap = 0.25                  # <--- this is the overlap in percent
    
    # --- }}}

    # --- init function --- {{{
    def __init__(self, path):
        """Load a MAT file.
        
        .. todo:
            check if the path is a file
        """
        
        # check if only one file is given
        if path.endswith(".mat"):
            realpath, matfile = os.path.split(path)
            self.path = os.path.abspath(realpath)
            self.mat_files = [self.path + "/" + matfile]
            if os.path.exists(self.mat_files[0]):
                self.n_files = 1
            else:
                self.n_files = 0
        
        # if a directory is given, so load all MAT files in it.
        else:
            self.path = path
            self.mat_files = glob.glob(self.path + "/*.mat")
            self.n_files = len(self.mat_files)
        
        if self.n_files == 0:
            raise ValueError("No valid MAT files were found")
    # --- }}}

    # --- load_from_mat --- {{{
    def load_from_mat(self, filename):
        """Load data from .MAT file.

        This function load the .MAT file provided after the RSI MATLAB
        procesing and creates a data structure in a dictionary.
        """

        # create empty dictionary to store data temporarily
        data = {}

        # load mat file
        mat = loadmat(filename)

        # list of variables
        # ---> in this variable i list the main varibales to
        #      extract from the MAT file. I skip the X_dX variables
        #      since I dont need them inmediately
        varlist = ["t_slow", "t_fast", "P_slow", "P_fast", "W_slow", "W_fast",
                   "Gnd", "Ax", "Ay", "Incl_X", "Incl_Y", "Incl_T", "V_Bat",
                   "T1_fast", "T1_slow", "gradT1", "sh1", "sh2",
                   "JAC_C", "JAC_T", "Turbidity", "Chlorophyll"]                   

        # load each variable and store in dict
        for var in varlist:
            data[var] = np.squeeze(mat[var])

        # compute datetime
        start_date_str = mat['date'][0] + ' ' + mat['time'][0]
        start_date = dt.datetime.strptime(start_date_str,
                                          "%Y-%m-%d %H:%M:%S.%f")
        data["start_date"] = start_date 

        # get sample frequencies
        data["fs_fast"] = mat["fs_fast"][0][0]
        data["fs_slow"] = mat["fs_slow"][0][0]

        # correct pressure if an offset exists
        data["P_fast"] -= self.P_offset
        data["P_slow"] -= self.P_offset

        # return data
        return data
    # --- }}}

    # --- load_single_file --- {{{
    def load_single_file(self, filename):
        """
        This function uses pickle to serialize the data object (dictionary) into
        a file. It is used for speed up the processing of the data. If the
        pickle doest not exist, the `load_from_mat` function is used.
        """
        # check if the pickle name exists
        pklname = os.path.splitext(filename)[0] + ".pkl"
        if os.path.exists(pklname):
            with open(pklname, "rb") as pklfile:
                data = pickle.load(pklfile)

        # if it doest not exist load it from matfile
        else:
            data = self.load_from_mat(filename)
            with open(pklname, "wb") as pklfile:
                pickle.dump(data, pklfile)

        return data
    # --- }}}

    # --- get_profiles --- {{{
    def get_profiles(self, min_duration=20, speed_cutoff=0.2, depth_cutoff=1.0):
        """This function detects the profiles in each dataset of timeseries.
        
        Args:
            min_duration: minumum duration of profile in seconds
            speed_cutoff: minumum fall velocity
            depth_cutoff: minimum depth

        .. todo:
            Make a way to get the data from cache instead of the run the code
            each time we want the data.
        """

        # initialize variables
        j = 1
        slow = {}
        fast = {}

        # for each file
        for filename in self.mat_files:

            # load data structure from mat or cache
            data = self.load_single_file(filename)
            
            # print in screen
            print(f"---> Loading file correponding to {data['start_date']}")

            # get pressure and fall speed
            P_slow = data["P_slow"]
            W_slow = data["W_slow"]

            # locate indices where the conditions are fulfilled
            ix = np.logical_and(P_slow > depth_cutoff, W_slow >= speed_cutoff).nonzero()[0]
            diff_ix = np.hstack((1, np.diff(ix)))
            m = (diff_ix > 1).nonzero()[0]

            # indices of slow channels
            ix_slow = [None] * len(m)
            ix_slow[0] = (ix[0], ix[m[1]-1])
            for i in range(1, len(m)-1):
                ix_slow[i] = (ix[m[i]], ix[m[i+1]-1])
            ix_slow[len(m)-1] = (ix[m[-1]], ix[-1])

            # indices of fast channels
            fac = np.int(np.squeeze(data['fs_fast']) / np.squeeze(data['fs_slow']))
            ix_fast = [(ix_slow[i][0]*fac, ix_slow[i][1]*fac) for i in range(len(ix_slow))]

            # numer of minimum sample and length of each profiles
            min_samples = int(min_duration * data["fs_slow"])

            # finally, store indices in arrays
            n_profiles = 0
            for i in range(len(m)):
                #
                lenght_profile = ix_slow[i][1] - ix_slow[i][0]
                #
                if lenght_profile > min_samples:
                    
                    # get start and end indices
                    i_slow, j_slow = ix_slow[i][0], ix_slow[i][1]
                    i_fast, j_fast = ix_fast[i][0], ix_fast[i][1]

                    # get starting and final date of profile
                    t0 = data["start_date"]
                    t_beg = t0 + dt.timedelta(seconds=data["t_slow"][i_slow])
                    t_end = t0 + dt.timedelta(seconds=data["t_slow"][j_slow])

                    # save slow channels
                    slow[f"t_{j:02d}"]      = data["t_slow"][i_slow:j_slow]
                    slow[f"P_{j:02d}"]      = data["P_slow"][i_slow:j_slow]
                    slow[f"W_{j:02d}"]      = data["W_slow"][i_slow:j_slow]
                    slow[f"C_{j:02d}"]      = data["JAC_C"][i_slow:j_slow]
                    slow[f"T_{j:02d}"]      = data["JAC_T"][i_slow:j_slow]
                    slow[f"Incl_X_{j:02d}"] = data["Incl_X"][i_slow:j_slow]
                    slow[f"Incl_Y_{j:02d}"] = data["Incl_Y"][i_slow:j_slow]

                    # save fast channels
                    fast[f"t_{j:02d}"]      = data["t_fast"][i_fast:j_fast]
                    fast[f"P_{j:02d}"]      = data["P_fast"][i_fast:j_fast]
                    fast[f"W_{j:02d}"]      = data["W_fast"][i_fast:j_fast]
                    fast[f"T_{j:02d}"]      = data["T1_fast"][i_fast:j_fast]
                    fast[f"sh1_{j:02d}"]    = data["sh1"][i_fast:j_fast]
                    fast[f"sh2_{j:02d}"]    = data["sh2"][i_fast:j_fast]
                    fast[f"tur_{j:02d}"]    = data["Turbidity"][i_fast:j_fast]
                    fast[f"chl_{j:02d}"]    = data["Chlorophyll"][i_fast:j_fast]
                    fast[f"Ax_{j:02d}"]     = data["Ax"][i_fast:j_fast]
                    fast[f"Ay_{j:02d}"]     = data["Ay"][i_fast:j_fast]

                    # save begining time
                    slow[f"date_{j:02d}"]  = t_beg
                    fast[f"date_{j:02d}"]  = t_beg

                    # count number or profiles
                    t_beg_s = t_beg.strftime("%X")
                    t_end_s = t_end.strftime("%X")
                    print(f"     Profile {j:02d} from {t_beg_s} to {t_end_s}")
                    j += 1

        # save sample frequency
        slow["fs"] = data["fs_slow"]
        fast["fs"] = data["fs_fast"]

        # store in main structure
        self.n_profiles = j - 1
        self.slow = slow
        self.fast = fast
    # --- }}}

    # --- extract_raw_profile --- {{{
    def extract_raw_profile(self, profile_number=1):
        """Extract time series of one profile_number.
        
        Args:
            profile_number: Number of the profile you want to retrieve
        """

        # check input
        if profile_number > self.n_profiles:
            raise Exception(f"There are only {self.n_profiles} profiles")
        elif profile_number < 1:
            raise Exception(f"Profiles start with 1")
        else:
            j = profile_number

        # create empty dictionary
        t = {}

        # extract date and time
        t["date"] = self.slow[f"date_{j:02d}"]

        # extract slow channels
        t["t_slow"] = self.slow[f"t_{j:02d}"]
        t["P_slow"] = self.slow[f"P_{j:02d}"]
        t["W_slow"] = self.slow[f"W_{j:02d}"]
        t["C_slow"] = self.slow[f"C_{j:02d}"]
        t["T_slow"] = self.slow[f"T_{j:02d}"]
        t["Incl_X"] = self.slow[f"Incl_X_{j:02d}"]
        t["Incl_Y"] = self.slow[f"Incl_Y_{j:02d}"]

        # extract fast channels
        t["t_fast"] = self.fast[f"t_{j:02d}"]
        t["P_fast"] = self.fast[f"P_{j:02d}"]
        t["W_fast"] = self.fast[f"W_{j:02d}"]
        t["T_fast"] = self.fast[f"T_{j:02d}"]
        t["sh1"]    = self.fast[f"sh1_{j:02d}"]
        t["sh2"]    = self.fast[f"sh2_{j:02d}"]
        t["tur"]    = self.fast[f"tur_{j:02d}"]
        t["chl"]    = self.fast[f"chl_{j:02d}"]
        t["Ax"]     = self.fast[f"Ax_{j:02d}"]
        t["Ay"]     = self.fast[f"Ay_{j:02d}"]

        # add other channels
        # compute some water properties
        sal, rho, kvisc = seawater(t["P_fast"], t["T_slow"], t["C_slow"])
        t["salt"] = sal
        t["rho"] = rho
        t["kvisc"] = kvisc

        # perform despiking of shear data
        t["sh1_despiked"] = despike(t["sh1"])
        t["sh2_despiked"] = despike(t["sh2"])

        # apply high pass filter to shear data
        for sh in ["sh1", "sh2"]:
            t[sh + "_clean"] = butterworth(
                                   t[sh + "_despiked"],
                                   self.fast["fs"],
                                   smooth=self.shear_smooth,
                                   kind="high"
                               )

        return t
    # --- }}}

    # --- extract_binned_profile --- {{{
    def extract_binned_profile(self, profile_number=1):
        """Extract binned variables for the time series.

        Args:
            profile_number: Number of the profile you want to retrieve

        Returns: The average of the following variables are going to be returned:
            * Depth: z
            * Rate of dissipation: e1, e2
            * Shear: sh1, sh2
            * Temperature: T
            * Salinity: S
            * Density: rho
            * Viscosity: nu
            * Brunt-Vaisalla: N2
        """
        # first extract time series
        j = profile_number
        t = self.extract_raw_profile(profile_number=j)

        # determine window length
        N = len(t["P_fast"])
        step = int(self.overlap * self.window_length)

        # start index for the first window
        P_min = self.P_min
        j_start = np.where(t["P_fast"] >= P_min)[0][0]

        # pre-define arrays arrays to save binned variables
        p = {}
        varlist = ["z", "e", "e1", "e2", "sh1", "sh2", "T", "S", "rho", "nu", "N2"]
        for var in varlist: 
            if var in ["z", "e", "e1", "e2"]:
                p[var] = np.array([])
            else:
                p[var+"_avg"] = np.array([])
                p[var+"_std"] = np.array([])

        # loop for each window
        count = 0
        spec_list = []
        for j in np.arange(j_start, N-j_start, step):
            
            # compute start and end indices and break if b>j_final
            a = j
            b = j + self.window_length
            if b >= N-self.window_length:
                break
            
            # compute brunt-vaisalla frequency
            N2 = gsw.Nsquared(
                                t["salt"][a:b],
                                t["T_fast"][a:b],
                                t["P_fast"][a:b],
                                lat=31
                             )[0]

            # compute cross-spectral matrices and corrected shear
            s = shear_spectrum(t, a, b, self.fft_params, return_all=True)
            k, P_sh1, P_sh2 = s["k"], s["P_sh1_clean"], s["P_sh2_clean"]
            
            # compute mean kinematic viscosity, mean fall speed and pressure
            nu = np.mean(t["kvisc"][a:b])
            Pmean = np.mean(t["P_fast"][a:b])
            Wmean = np.mean(t["W_fast"][a:b])
            
            # compute dissipation rate using RSI algorithm
            epsilon1, kmax1 = get_dissrate(k, P_sh1, Wmean, nu)
            epsilon2, kmax2 = get_dissrate(k, P_sh2, Wmean, nu)

            # add epsilon and kmax to spectra dictionary
            s["e1"], s["e2"] = epsilon1, epsilon2
            s["kmax1"], s["kmax2"] =  kmax1, kmax2

            # add minimum and maximun depths to the spectrum dictionay
            s["z_min"] = t["P_fast"][a]
            s["z_max"] = t["P_fast"][b]
            
            # concatenate variables
            p["z"] = np.append(p["z"], -Pmean)
            #
            p["e1"] = np.append(p["e1"], epsilon1)
            p["e2"] = np.append(p["e2"], epsilon2)
            p["e"]  = np.append(p["e"], 0.5 * (epsilon1 + epsilon2))
            #
            p["sh1_avg"] = np.append(p["sh1_avg"], np.nanmean(t["sh1_clean"][a:b]))
            p["sh2_avg"] = np.append(p["sh2_avg"], np.nanmean(t["sh2_clean"][a:b]))
            p["sh1_std"] = np.append(p["sh1_std"], np.nanstd(t["sh1_clean"][a:b]))
            p["sh2_std"] = np.append(p["sh2_std"], np.nanstd(t["sh2_clean"][a:b]))
            #
            p["T_avg"] = np.append(p["T_avg"], np.nanmean(t["T_fast"][a:b]))
            p["T_std"] = np.append(p["T_std"], np.nanstd(t["T_fast"][a:b]))
            #
            p["S_avg"] = np.append(p["S_avg"], np.nanmean(t["salt"][a:b]))
            p["S_std"] = np.append(p["S_std"], np.nanstd(t["salt"][a:b]))
            #
            p["rho_avg"] = np.append(p["rho_avg"], np.nanmean(t["rho"][a:b]))
            p["rho_std"] = np.append(p["rho_std"], np.nanstd(t["rho"][a:b]))
            #
            p["nu_avg"] = np.append(p["nu_avg"], np.nanmean(t["kvisc"][a:b]))
            p["nu_std"] = np.append(p["nu_std"], np.nanstd(t["kvisc"][a:b]))
            #
            p["N2_avg"] = np.append(p["N2_avg"], np.nanmean(N2))
            p["N2_std"] = np.append(p["N2_std"], np.nanstd(N2))
            #
            spec_list.append(s)

            # update counter
            count += 1
        
        # add list of spectra and number of segments
        p["spectra"] = spec_list
        p["number_of_segments"] = count - 1

        # add date to the dict
        p["date"] = t["date"]

        #return data
        return p
    # --- }}}

    # --- profile --- {{{
    def profile(self, profile_number=1):
        """Load profile data from cache if exists.

        Args:
            profile_number: Number of the profile you want to get.

        Returns:
            Dictionary with all the info about the profile.

        """

        # get the date of the profile
        j = profile_number
        fmt = "%Y%m%d%H%M%S"
        date = self.slow[f"date_{j:02d}"].strftime(fmt)
        
        # check if the cache file exists
        pklname = self.cachepath + f"/{date}.pkl" 
        if os.path.exists(pklname):
            with open(pklname, "rb") as pklfile:
                p = pickle.load(pklfile)

        # if it doest not exist load run the function and store in cache
        else:
            p = self.extract_binned_profile(profile_number)
            
            with open(pklname, "wb") as pklfile:
                pickle.dump(p, pklfile)
        
        # finally return data
        return p



    # --- }}}

# --- end of main class --- }}}



# --- quick view class ---- {{{
class QuickView(object):

    """
    This class contains script for plotting and analysing main data of the VMP
    profiles.
    """

    # --- init function --- {{{
    def __init__(self, vmp, profile_number):
        
        # number of profile
        self.profile_number = profile_number

        # get profiles and data from a given Vmp instance
        try:
            self.t = vmp.extract_raw_profile(self.profile_number)
            self.p = vmp.extract_binned_profile(self.profile_number)
        except AttributeError:
            vmp.get_profiles()
            self.t = vmp.extract_raw_profile(self.profile_number)
            self.p = vmp.extract_binned_profile(self.profile_number)

        # prepare folders
        print("")
        print(f"---> Creating folder for profile {self.profile_number:02d}")
        os.system(f"mkdir -p figs/profile_{profile_number:02d}/despike")
        os.system(f"mkdir -p figs/profile_{profile_number:02d}/spectra")
        os.system(f"mkdir -p figs/profile_{profile_number:02d}/profiles")

    # --- }}}

    # --- epsilon_profile --- {{{
    def epsilon_profile(self):
        """
        """

        # extract some class info to local level
        t = self.t
        p = self.p
        j = self.profile_number

        # make figure
        fig = plt.figure(figsize=(7.2,4.2))
        ax1 = fig.add_subplot(1,5,(1,2))
        ax2 = fig.add_subplot(1,5,3)
        ax3 = fig.add_subplot(1,5,4, sharey=ax2)
        ax4 = fig.add_subplot(1,5,5, sharey=ax2)
        #
        e = 0.5*(p["e1"] + p["e2"])
        ax1.semilogx(p["e1"], p["z"], '.-',  label="Probe 1")
        ax1.semilogx(p["e2"], p["z"], '.-',  label="Probe 2")
        ax1.semilogx(e,  p["z"], '.-k')
        ax1.legend(loc=3, ncol=2)
        ax1.set_ylim((-100,0))
        ax1.set_xlim((1E-10,1E-4))
        ax1.set_xlabel("$\epsilon\,\mathrm{[W\;kg^{-1}]}$")
        ax1.set_ylabel("$z\,\mathrm{[m]}$")
        ax1.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        ax2.plot(t["sh1_clean"]-1, -t["P_fast"])
        ax2.plot(t["sh2_clean"]+1, -t["P_fast"])
        ax2.set_ylim((-100,0))
        ax2.set_xlim((-3.5,3.5))
        ax2.set_yticklabels([])
        ax2.set_xlabel("$\partial u / \partial z\;\mathrm{[s^{-1}]}$")
        ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        T = t["T_slow"] # <--- store temperature in T for facility
        Th = (T - np.sort(T)[::-1]) * 100
        ax3.plot(Th, -t["P_slow"], c='r', alpha=0.2)
        ax3.plot(butterworth(Th, fs=64, smooth=1), -t["P_slow"], c='r', alpha=.5)
        ax3.set_xlabel("$T\,\mathrm{[-]}$")
        ax3.set_ylim((-100,0))
        ax3b = ax3.twiny()
        ax3b.plot(T, -t["P_slow"], lw=1.5, c='r')
        ax3b.set_xticklabels([])
        ax3b.set_ylim((-100,0))
        ax3b.margins(.2)
        #
        W = t["W_slow"]
        ax4.plot((W-W.mean()) / W.std(), -t["P_slow"], lw=1.5, c='b')
        ax4.plot(butterworth(t["Ax"], fs=512)/5, -t["P_fast"], c='g')
        ax4.plot(butterworth(t["Ay"], fs=512)/5, -t["P_fast"], c='r')
        ax4.set_xlabel("$W$, $A_x$, $A_y$, $\mathrm{[-]}$")
        ax4.set_xlim((-5.5,5.5))
        ax4.set_ylim((-100,0))
        ax4.margins(.2)
        #
        fig.suptitle(f"Profile {j:02d}\n{t['date'].strftime('%Y-%m-%d %H:%M')} UTC")
        fig.subplots_adjust(top=.9, bottom=.12, right=.95, left=.10)
        fig.savefig(f'figs/profile_{j:02d}/profiles/epsilon_{j:02d}.pdf')
        plt.close("all")



    # --- }}}

    # --- biological_profile --- {{{
    def biological_profile(self):
        """
        """

        # extract some class info to local level
        t = self.t
        j = self.profile_number

        # make figure
        fig = plt.figure(figsize=(7.2,4.2))
        ax1 = fig.add_subplot(1,5,1)
        ax2 = fig.add_subplot(1,5,2)
        ax3 = fig.add_subplot(1,5,3, sharey=ax2)
        ax4 = fig.add_subplot(1,5,4, sharey=ax2)
        ax5 = fig.add_subplot(1,5,5, sharey=ax2)
        #
        T = t["T_slow"]
        ax1.plot(T, -t["P_slow"], c='r', alpha=0.2)
        ax1.plot(butterworth(T, fs=64), -t["P_slow"], c='r', lw=1.5)
        ax1.set_ylim((-100,0))
        ax1.set_xlabel("$\mathrm{Temperature\;[{}^\circ{}C]}$")
        ax1.set_ylabel("$z\,\mathrm{[m]}$")
        # ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
        # ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax1.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        S = despike(t["salt"])
        ax2.plot(S, -t["P_fast"], c='#0080FF', alpha=0.2)
        ax2.plot(butterworth(S, fs=512), -t["P_fast"], c='#0080FF')
        ax2.set_xlabel("$\mathrm{Salinity\;[ppm]}$")
        ax2.set_ylim((-100,0))
        ax2.set_yticklabels([])
        ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        rho = despike(t["rho"])
        ax3.plot(rho, -t["P_fast"], c='#FF8000', alpha=0.2)
        ax3.plot(butterworth(rho, fs=512), -t["P_fast"], c='#FF8000')
        ax3.set_xlabel("$\mathrm{Density\;[kg\,m^{-3}]}$")
        ax3.set_ylim((-100,0))
        ax3.set_yticklabels([])
        ax3.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax3.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        chl = despike(t["chl"])
        ax4.plot(chl, -t["P_fast"], c='#4B8A08', alpha=0.2)
        ax4.plot(butterworth(chl, fs=512), -t["P_fast"], c='#4B8A08')
        ax4.set_ylim((-100,0))
        ax4.set_xlabel("$\mathrm{Chlorophyll\;[ppb]}$")
        ax4.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax4.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        tur = despike(t["tur"], threshold=10)
        ax5.plot(tur, -t["P_fast"], c='#3A2F0B', alpha=0.2)
        ax5.plot(butterworth(tur, fs=512), -t["P_fast"], c='#3A2F0B')
        ax5.set_ylim((-100,0))
        ax5.set_xlabel("$\mathrm{Turbidity\;[FTU]}$")
        ax5.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax5.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        fig.suptitle(f"Profile {j:02d}\n{t['date'].strftime('%Y-%m-%d %H:%M')} UTC")
        fig.subplots_adjust(top=.9, bottom=.12, right=.95, left=.10)
        fig.savefig(f'figs/profile_{j:02d}/profiles/biological_{j:02d}.pdf')
        plt.close("all")



    # --- }}}

    # --- acceleration_profile --- {{{
    def acceleration_profile(self):
        """
        """

        # extract some class info to local level
        t = self.t
        j = self.profile_number

        # make figure
        fig = plt.figure(figsize=(7.2,4.2))
        ax1 = fig.add_subplot(1,4,1)
        ax2 = fig.add_subplot(1,4,2)
        ax3 = fig.add_subplot(1,4,3, sharey=ax2)
        ax4 = fig.add_subplot(1,4,4, sharey=ax2)
        #
        Ax = despike(t["Ax"], fs=512, threshold=10)
        ax1.plot(Ax, -t["P_fast"], c='k', alpha=0.2)
        ax1.plot(butterworth(Ax, fs=512), -t["P_fast"], c='k', lw=1.5)
        ax1.set_ylim((-100,0))
        ax1.set_xlabel("$\mathrm{Acceleration\;X\;[m\;s^{-2}]}$")
        ax1.set_ylabel("$z\,\mathrm{[m]}$")
        ax1.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        Ay = despike(t["Ay"], fs=512, threshold=10)
        ax2.plot(Ay, -t["P_fast"], c='k', alpha=0.2)
        ax2.plot(butterworth(Ay, fs=512), -t["P_fast"], c='k', lw=1.5)
        ax2.set_ylim((-100,0))
        ax2.set_yticklabels([])
        ax2.set_xlabel("$\mathrm{Acceleration\;Y\;[m\;s^{-2}]}$")
        ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax2.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        Incl_X = despike(t["Incl_X"], fs=64, threshold=10)
        ax3.plot(Incl_X, -t["P_slow"], c='k', alpha=0.2)
        ax3.plot(butterworth(Incl_X, fs=64), -t["P_slow"], c='k', lw=1.5)
        ax3.set_ylim((-100,0))
        ax3.set_yticklabels([])
        ax3.set_xlabel("$\mathrm{Inclination\;X\;[{}^\circ{}]}$")
        ax3.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax3.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        Incl_Y = despike(t["Incl_Y"], fs=64, threshold=10)
        ax4.plot(Incl_Y, -t["P_slow"], c='k', alpha=0.2)
        ax4.plot(butterworth(Incl_Y, fs=64), -t["P_slow"], c='k', lw=1.5)
        ax4.set_ylim((-100,0))
        ax4.set_yticklabels([])
        ax4.set_xlabel("$\mathrm{Inclination\;Y\;[{}^\circ{}]}$")
        ax4.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax4.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
        #
        fig.suptitle(f"Profile {j:02d}\n{t['date'].strftime('%Y-%m-%d %H:%M')} UTC")
        fig.subplots_adjust(top=.9, bottom=.12, right=.95, left=.10)
        fig.savefig(f'figs/profile_{j:02d}/profiles/acceleration_{j:02d}.pdf')
        plt.close("all")



    # --- }}}

    # --- spectrum correction --- {{{
    def spectrum_correction(self):
        """
        """

        # extract some class info to the local level
        j = self.profile_number
        p  = self.p

        # loop for each window segment along the profile
        for i, s in enumerate(self.p["spectra"]):
            #
            # print in screen
            print(f" - Computing spectra segment: {s['z_min']:.2f} - {s['z_max']:.2f} m")
            print(f"     Degrees of freedom n = {s['dof']}")
            print(f"     Correction of vibration = {s['correction_status']}")
            print(f"     Dissipation rate 1 = {p['e1'][i]:.2E} W/kg")
            print(f"     Dissipation rate 2 = {p['e2'][i]:.2E} W/kg")
            #
            fig = plt.figure(figsize=(6.2, 3.1))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
            #
            for e in np.logspace(-9, -3, 10):
                ax1.loglog(s["k"], nasmyth(k=s["k"], epsilon=e), c='0.8')
                ax2.loglog(s["k"], nasmyth(k=s["k"], epsilon=e), c='0.8')
            #
            ks1 = (p['e1'][i]/p["nu_avg"][i] ** 3)**(1./4.)
            ks2 = (p['e2'][i]/p["nu_avg"][i] ** 3)**(1./4.)
            #
            ax1.loglog(s["k"], nasmyth(k=s["k"], epsilon=p["e1"][i]), c='0.5')
            ax2.loglog(s["k"], nasmyth(k=s["k"], epsilon=p["e2"][i]), c='0.5')
            #
            ax1.loglog(ks1, nasmyth(k=ks1, epsilon=p["e1"][i]), '.', c='0.5')
            ax2.loglog(ks1, nasmyth(k=ks1, epsilon=p["e2"][i]), '.', c='0.5')
            #
            ax1.loglog(s["k"], s["P_Ax"]/1E3, '--')
            ax1.loglog(s["k"], s["P_Ay"]/1E3, '--')
            ax1.loglog(s["k"], s["P_sh1"], c="r")
            ax1.loglog(s["k"], s["P_sh1_clean"], "k", lw=1.0)
            ax1.set_xlabel("$k\,\mathrm{[cpm]}$")
            ax1.set_ylabel("$S\,\mathrm{[units^2/cpm]}$")
            #
            ax2.loglog(s["k"], s["P_Ax"]/1E3, '--')
            ax2.loglog(s["k"], s["P_Ay"]/1E3, '--')
            ax2.loglog(s["k"], s["P_sh2"], c="r")
            ax2.loglog(s["k"], s["P_sh2_clean"], "k", lw=1.0)
            ax2.set_xlabel("$k\,\mathrm{[cpm]}$")
            #
            suptitle = f"Profile {j:02d}\n" + \
                       f"${-s['z_min']:.2f}$ m $<z<-{s['z_max']:.2f}$ m; " + \
                       f"$W = {s['Wmean']:.2f}$ m/s; " + \
                       f"$P = {s['Pmean']:.2f}$ dbar\n" + \
                       f"$\epsilon_1 = {scientific_notation(p['e1'][i])}$ W/kg; " + \
                       f"$\epsilon_2 = {scientific_notation(p['e2'][i])}$ W/kg"
            fig.suptitle(suptitle)
            fig.subplots_adjust(top=.77, bottom=.15)
            fig.savefig(f'figs/profile_{j:02d}/spectra/shear{j:02d}_{i:02d}.pdf')
            plt.close("all")
    # --- }}}

# --- ends of class --- }}}


if __name__ == "__main__":
    pass

# --- end of file ---
