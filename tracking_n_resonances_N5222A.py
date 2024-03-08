import pyvisa as visa
import numpy as np
import pandas as pd
import time
from scipy.optimize import curve_fit
from datetime import datetime
from matplotlib import pyplot as plt
trace_parameter = {
    'S11': "'CH1_S11_2'",
    'S22': "'CH1_S22_3'",
    'S21': "'CH1_S21_1'",
}


def lorentzian(f, f0, df, A0, A1, B0, B1):
    return A0 + A1 * (f - f0) + (B0 + B1 * (f - f0)) / (1 + 4 * ((f - f0) / df) ** 2)


def inverselorentzian(f, f0, A0, A1, beta, Ql):
    delta = f / f0 - f0 / f
    return (A0 + A1 * (f - f0)) * (
        1 - (4 * beta) / ((1 + beta) ** 2 + (Ql * delta) ** 2))

#def linear(f, A0, A1)
#    return A0 + A1 * f


def connect_devices(devices):
    rm = visa.ResourceManager()
    for name, device in devices.items():
        if device['module'] == 'visa':
            device['instrument'] = rm.open_resource(
                device['address'])
        if name == 'networkAnalyzer':
            device['instrument'].write('FORMat:DATA ASCii')
            device['instrument'].write('INITiate:CONTinuous OFF')
            sweep_time = float(
                device['instrument'].query('SENSe:SWEep:TIME?'))


def disconnect_devices(devices):
    for name, device in devices.items():
        if name == 'networkAnalyzer':
            device['instrument'].write('INITiate:CONTinuous ON')
        if device['module'] == 'visa':
            device['instrument'].close()


def query_temperature(devices):
    instr = devices['temperatureController']['instrument']
    temps = np.array(
        [float(x) for x in
         instr.query('KRDG? 0').strip().split(',')])
    return temps


def initiate_networkAnalyzer(devices):
    instr = devices['networkAnalyzer']['instrument']
    sweep_time = float(instr.query('SENSe:SWEep:TIME?'))
    instr.write('INITiate')
    time.sleep(2 * sweep_time + 0.1)


def track_networkAnalyzer(devices, center, span):
    instr = devices['networkAnalyzer']['instrument']
    instr.write('SENSe:FREQuency:CENTer {}'.format(center))
    instr.write('SENSe:FREQuency:SPAN {}'.format(span))


def query_bandwidth(devices):
    instr = devices['networkAnalyzer']['instrument']
    param = trace_parameter['S21']
    instr.write(f'CALCulate:PARameter:SELect {param}')
    bwids = np.array(
        [float(x) for x in
         instr.query(f'CALCulate:MARKer1:BWIDth?').strip().split(',')])
    return bwids


def query_trace(devices):
    instr = devices['networkAnalyzer']['instrument']
    traces = {x: np.array([]) for x in trace_parameter}
    for trace, param in trace_parameter.items():
        instr.write(f'CALCulate:PARameter:SELect {param}')
        traces[trace] = np.array(
            [float(x) for x in instr.query('CALCulate:DATA? FDATA').\
             strip().split(',')])#[::2]

    sweep_points = int(instr.query('SENSe:SWEep:POINts?'))
    freq_start = float(instr.query('SENSe:FREQuency:STARt?'))
    freq_stop = float(instr.query('SENSe:FREQuency:STOP?'))
    freq = np.linspace(freq_start, freq_stop, sweep_points)

    return freq, traces


def fit_lorentzian(freq, traces, bwids):
    transmission = np.power(10, traces['S21'] / 10)
    p0 = [bwids[1], bwids[0],
          np.min(transmission),
          (transmission[-1] - transmission[0]) / (freq[-1] - freq[0]),
          np.max(transmission) - np.min(transmission), 0]

    popt, pcov = curve_fit(
        lorentzian, freq, transmission, p0=p0)

    return popt


def fit_inverselorentzian(freq, traces, sparam, bwids):
    reflection = np.power(10, traces[sparam] / 10)
    depth = np.min(traces[sparam]) - np.max(traces[sparam])
    beta0 = (1 - np.power(10, depth / 20)) / (1 + np.power(10, depth / 20))
    p0 = [bwids[0], np.max(reflection),
          (reflection[-1] - reflection[0]) / (freq[-1] - freq[0]),
          beta0, bwids[0] / bwids[1]]

    popt, pcov = curve_fit(
        inverselorentzian, freq, reflection, p0=p0)

    #print(popt)
    """
    fig, ax = plt.subplots()
    ax.plot(freq, reflection)
    ax.plot(freq, inverselorentzian(freq, *popt))
    plt.show()
    """

    return popt

#def fit_linear(freq, traces, sparam, bwids):
#    reflection = np.power(10, traces[sparam] / 10)
#    depth = np.min(traces[sparam]) - np.max(traces[sparam])
#    beta0 = (1 - np.power(10, depth / 20)) / (1 + np.power(10, depth / 20))
#    p0 = [bwids[0], np.max(reflection),
#          (reflection[-1] - reflection[0]) / (freq[-1] - freq[0]),
#          beta0, bwids[0] / bwids[1]]

#   popt, pcov = curve_fit(
#        linear, freq, reflection, p0=p0)
    """
    fig, ax = plt.subplots()
    ax.plot(freq, reflection)
    ax.plot(freq, inverselorentzian(freq, *popt))
    plt.show()
    """

#    return popt


def main():
    init_time = datetime.now()
    devices = {
        'temperatureController': {
            'module': 'visa',
            'address': 'GPIB1::12::INSTR',
        },
        'networkAnalyzer': {
            'module': 'visa',
            'address': 'TCPIP0::A-N5232A-21381::inst0::INSTR'
            #'address': 'GPIB0::16::INSTR',
        },
    }
    resonances = {
        #'Mode1': (5.8534e9, 4e6), # modify frequencies
        #'Mode2': (7.4791e9, 4e6)
        #'Mode1': (9.0787e9, 2e6),
        
        #'Mode1': (6.081587e9, 1e6),
        #'Mode2': (6.51567e9, 1e6),
        #'Mode3': (7.6275e9, 1e6)

        #'Mode1': (10.376e9, 5e6), # modify frequencies
        #'Mode2': (10.559e9, 5e6),
        #'Mode3': (10.860e9, 5e6),
        #'Mode4': (11.277e9, 5e6),
        #'Mode5': (11.791e9, 5e6)

        #'Mode1': (7.9501e9, 1e6) # modify frequencies RutileF1
        #'Mode2': (8.4403e9, 1e6)
        #'Mode3': (8.4503e9, 1e6),
        #'Mode4': (8.6935e9, 1e6),
        #'Mode5': (8.8486e9, 1e6),
        #'Mode6': (8.8685e9, 1e6),
        #'Mode7': (9.0290e9, 1e6),
        #'Mode8': (9.0467e9, 1e6),
        #'Mode9': (10.262e9, 1e6),
        #'Mode10': (10.270e9, 1e6),
        #'Mode11': (10.272e9, 1e6),
        #'Mode12': (10.276e9, 1e6),
        #'Mode13': (10.571e9, 1e6),
        #'Mode14': (10.726e9, 1e6),
        #'Mode15': (10.731e9, 1e6),
        #'Mode16': (10.925e9, 1e6),
        #'Mode17': (10.948e9, 1e6)

        #'Mode1': (7.9462e9, 1e6), # modify frequencies Rutile F2
        #'Mode2': (8.3709e9, 1e6),
        ##'Mode3': (8.383e9, 1e6),
        #'Mode3': (8.4302e9, 1e6),
        #'Mode4': (8.7891e9, 1e6),
        ##'Mode6': (8.908e9, 1e6),
        ##'Mode6': (8.9437e9, 1e6),
        #'Mode5': (10.198e9, 1e6),
        ##'Mode8': (10.203e9, 1e6),
        ##'Mode9': (10.209e9, 1e6),
        #'Mode6': (10.231e9, 1e6),
        ##'Mode7': (10.447e9, 1e6),
        ##'Mode8': (10.517e9, 1e6),
        ##'Mode9': (10.641e9, 1e6),
        ##'Mode10': (10.649e9, 1e6),
        #'Mode7': (10.858e9, 1e6)
        ##'Mode11': (10.074e9, 1e6),
        ##'Mode12': (11.086e9, 1e6),
        ##'Mode13': (11.210e9, 1e6)

        #'Mode1': (7.6732e9, 5e6), # modify frequencies
        #'Mode2': (8.8440e9, 5e6),
        #'Mode3': (9.2094e9, 5e6),
        #'Mode2': (9.2130e9, 5e6),
        #'Mode5': (9.5422e9, 5e6),
        #'Mode6': (10.446e9, 5e6),
        #'Mode7': (10.650e9, 5e6),
        #'Mode8': (10.789e9, 5e6),
        #'Mode3': (10.793e9, 5e6)
        #'Mode10': (10.831e9, 5e6),
        #'Mode11': (10.868e9, 5e6)

        'Mode1': (5.4634e9, 10e6),
        'Mode2': (5.4966e9, 10e6),
        'Mode3': (5.5496e9, 10e6),
        'Mode4': (9.4781e9, 10e6),
        'Mode5': (9.4842e9, 10e6),
        'Mode6': (9.4964e9, 10e6),
        'Mode7': (11.173e9, 10e6),
        'Mode8': (11.198e9, 5e6),
        'Mode9': (11.206e9, 5e6),
        'Mode10': (11.221e9, 10e6),
        'Mode11': (13.583e9, 10e6),
        'Mode12': (13.595e9, 10e6),
        'Mode13': (13.616e9, 10e6),
        'Mode14': (13.696e9, 10e6),
        'Mode15': (13.713e9, 10e6)

        #'Mode1': (9.4784e9, 2e6), # modify frequencies
        #'Mode2': (9.4841e9, 2e6),
        #'Mode3': (9.4965e9, 2e6)

        #'Mode1': (7.0738e9, 10e6), # modify frequencies
        #'Mode2': (7.2804e9, 10e6),
        #'Mode3': (7.8217e9, 10e6),
        #'Mode4': (8.0092e9, 10e6)

        #'Mode1': (7.0495e9, 10e6),
        #'Mode2': (7.2561e9, 10e6),
        #'Mode3': (7.7931e9, 10e6),
        #'Mode4': (7.9812e9, 10e6)

        #'Mode1': (5.4061e9, 50e6),
        #'Mode2': (5.6462e9, 50e6)
        #'Mode1': (5.4060e9, 20e6),
        #'Mode2': (5.6461e9, 20e6)
        #'Mode1': (1.0387e9, 20e6)

        #'Mode1': (10.643e9, 30e6), # modify frequencies
        #'Mode2': (10.829e9, 30e6),
        #'Mode3': (11.129e9, 30e6),
        #'Mode4': (11.537e9, 30e6),
        #'Mode5': (12.044e9, 30e6)

        #'Mode1': (2.3074e9, 10e6),
        #'Mode2': (2.4760e9, 10e6),
        #'Mode3': (2.8402e9, 10e6),
        #'Mode4': (3.3497e9, 10e6),
        #'Mode5': (3.9445e9, 10e6)
    }
    coupling = 0 # Set coupling reading status 1: on / 0: off
    zoomout = 0 #widespan 1: on / 0: off
    widespan = 10e6
    data = {
        'time': np.array([]),
        'temp1': np.array([]),
        'temp2': np.array([]),
        'temp3': np.array([]),
        'temp4': np.array([]),
    }
    for mode in resonances:
        data[mode + '_f0'] = np.array([])
        data[mode + '_ql'] = np.array([])
        data[mode + '_c1'] = np.array([])
        data[mode + '_c2'] = np.array([])
    bwids = {}
    betas = {}

    connect_devices(devices)
    idx = 0
    while True:
        try:
            current_time = datetime.now()
            elapsed_time = (current_time - init_time).total_seconds()
            temps = query_temperature(devices)
            for mode, (f0, df) in resonances.items():
                if zoomout == 0:
                    track_networkAnalyzer(devices, f0, 5 * df)
                    initiate_networkAnalyzer(devices)
                elif zoomout == 1:
                    track_networkAnalyzer(devices, f0, widespan)
                    initiate_networkAnalyzer(devices)
                    df1, f1, qfactor, loss  = query_bandwidth(devices)
                    track_networkAnalyzer(devices, f1, 5 * df1)
                else:
                    print('Wide check error')
                initiate_networkAnalyzer(devices)
                marker_bwids = query_bandwidth(devices)
                freq, traces = query_trace(devices)
                bwids[mode] = fit_lorentzian(freq, traces, marker_bwids)
                betas[mode] = [(), ()]
                
                if coupling == 0:
                    betas[mode][0] = [0,0,0,0,0]
                    betas[mode][1] = [0,0,0,0,0]
                elif coupling == 1:
                    try:
                        betas[mode][0] = fit_inverselorentzian(
                        freq, traces, 'S11', bwids[mode])
                        #betas[mode][0] = [0,0,0,0,0]
                    except:
                        betas[mode][0] = [0,0,0,0,0]
                        continue
                    try:
                        betas[mode][1] = fit_inverselorentzian(
                        freq, traces, 'S22', bwids[mode])
                        #betas[mode][0] = [0,0,0,0,0]
                    except:
                        betas[mode][1] = [0,0,0,0,0]
                        continue
                     
                    #betas[mode][1] = [0,0,0,0,0] 
                else:
                    print("Coupling Error")
                
                resonances[mode] = (bwids[mode][0], bwids[mode][1])
                #input(mode)

            idx += 1

            data['time'] = np.append(data['time'], elapsed_time)
            data['temp1'] = np.append(data['temp1'], temps[4])
            data['temp2'] = np.append(data['temp2'], temps[5])
            data['temp3'] = np.append(data['temp3'], temps[6])
            data['temp4'] = np.append(data['temp4'], temps[7])
            print('[{}] T1 {:.2f} K, T50K {:.2f} K, T2 {:.2f}, T4K {:.2f}'.format(
                current_time.strftime('%y/%m/%d %H:%M:%S'), temps[4], temps[5], temps[6], temps[7]))

            for mode in resonances:
                data[mode + '_f0'] = np.append(
                    data[mode + '_f0'], bwids[mode][0])
                data[mode + '_ql'] = np.append(
                    data[mode + '_ql'], bwids[mode][0] / bwids[mode][1])
                data[mode + '_c1'] = np.append(
                    data[mode + '_c1'], betas[mode][0][3])
                data[mode + '_c2'] = np.append(
                    data[mode + '_c2'], betas[mode][1][3])

                print(mode)
                print('F {:5f} GHz, Q {:.0f}, c1 {:.2e}, c2 {:.2e}'.format(
                    bwids[mode][0] / 1e9, bwids[mode][0] / bwids[mode][1],
                    betas[mode][0][3], betas[mode][1][3]))

            if idx % 10 == 0:
                df = pd.DataFrame(data)
                df.to_csv('data/{}_tracking.csv'.format(
                    init_time.strftime('%y%m%d_%H%M%S')), index=False)

        except KeyboardInterrupt:
            break
        except:
            continue
    df = pd.DataFrame(data)
    #df.to_csv('data/{}_tracking_RutileF1_4-300K.csv'.format(
    df.to_csv('data/{}_tracking_TestCavityTM5.5GHz_LAntenna_300K_without_coupling.csv'.format(
    #df.to_csv('data/{}_tracking_TwoCellCopper03_300-4K_without_coupling.csv'.format(
    #df.to_csv('data/{}_tracking_EuBCO02Cavity_Shielded_4-300K_without_coupling.csv'.format(
    #df.to_csv('data/{}_tracking_8TBCavity_2Alumina_4-300K_without_coupling.csv'.format(
        init_time.strftime('%y%m%d_%H%M%S')), index=False)
    disconnect_devices(devices)


if __name__ == '__main__':
    main()

# SmallCuCavity