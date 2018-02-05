from libc.math cimport exp, sqrt, log, log10, cos, abs

def chem_solver(double dt, double HOUR, double TEMP, double C_M, double HEIGHT, double conc_O3, double conc_NO2, double conc_NO, double conc_HNO3, double conc_HO):
    cdef double dconc_O3 = dO3_dt( HOUR,  TEMP,  C_M,  HEIGHT,  conc_NO2,  conc_NO,  conc_O3)
    cdef double dconc_NO2 = dNO2_dt( HOUR,  TEMP,  C_M,  HEIGHT,  conc_NO2,  conc_NO,  conc_O3,  conc_HO)
    cdef double dconc_NO = dNO_dt( HOUR,  TEMP,  C_M,  HEIGHT,  conc_NO2,  conc_NO,  conc_O3)
    cdef double dconc_HNO3 = dHNO3_dt( HOUR,  TEMP,  C_M,  HEIGHT,  conc_NO2,  conc_HO)
    cdef double dconc_HO = dHO_dt( TEMP,  C_M)
    return { "conc_O3": dconc_O3, "conc_NO2": dconc_NO2, "conc_NO": dconc_NO, "conc_HNO3": dconc_HNO3, "conc_HO": dconc_HO }

cdef double dO3_dt(double HOUR, double TEMP, double C_M, double HEIGHT, double conc_NO2, double conc_NO, double conc_O3):
    cdef double dO3 = -1.0*ARR2( 1.40e-12 , 1310, TEMP )*conc_NO*conc_O3 + 1.0*0.01 *conc_NO2;
    return dO3


cdef double dNO2_dt(double HOUR, double TEMP, double C_M, double HEIGHT, double conc_NO2, double conc_NO, double conc_O3, double conc_HO):
    cdef double dNO2 = 1.0*ARR2( 1.40e-12 , 1310, TEMP )*conc_NO*conc_O3 + -1.0*0.01 *conc_NO2 + -1.0*TROE( 1.49e-30 , 1.8 , 2.58e-11 , 0 , TEMP, C_M)*conc_NO2*conc_HO;
    return dNO2


cdef double dNO_dt(double HOUR, double TEMP, double C_M, double HEIGHT, double conc_NO2, double conc_NO, double conc_O3):
    cdef double dNO = -1.0*ARR2( 1.40e-12 , 1310, TEMP )*conc_NO*conc_O3 + 1.0*0.01 *conc_NO2;
    return dNO


cdef double dHNO3_dt(double HOUR, double TEMP, double C_M, double HEIGHT, double conc_NO2, double conc_HO):
    cdef double dHNO3 = 1.0*TROE( 1.49e-30 , 1.8 , 2.58e-11 , 0 , TEMP, C_M)*conc_NO2*conc_HO;
    return dHNO3


cdef double dHO_dt(double TEMP, double C_M):
    cdef double dHO = 0;
    return dHO


####################
# RATE EXPRESSIONS #
####################

cdef ARR2(double A0, double B0, double TEMP):
    return A0 * exp(-B0 / TEMP)


cdef TROE(double k0_300K, double n, double kinf_300K, double m, double TEMP, double C_M):
    cdef double zt_help;
    cdef double k0_T;
    cdef double kinf_T;
    cdef double k_ratio;
    zt_help = 300.0 / TEMP;
    k0_T    = k0_300K   * zt_help ** n * C_M;
    kinf_T  = kinf_300K * zt_help ** m;
    k_ratio = k0_T/kinf_T;
    return k0_T/(1.0 + k_ratio)*0.6 ** (1.0 / (1.0+log10(k_ratio)**2))
