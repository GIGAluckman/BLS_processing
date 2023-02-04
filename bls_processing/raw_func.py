import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Slider

class BLSfile_raw():
    def __init__(self, filemask):
        self.file = h5py.File(filemask, 'r')
        
    def acquire_spectrum(self):
        
        for index,row in enumerate(self.file['scan_definition'].keys()):
            tvar = self.file['scan_definition'][row][1,1]
            if tvar == b'Acquire spectrum':
                dataRow = row
        
        main_data = self.file['measurement'][dataRow]['data']
        num_int_freq_points = main_data.shape[1]
        left_int_freq_lim = self.file['measurement'][dataRow]['scale'][0]
        right_int_freq_lim_f = self.file['measurement'][dataRow]['scale'][1]
        right_int_freq_lim = left_int_freq_lim + right_int_freq_lim_f * (num_int_freq_points-1)
        
        return np.array(main_data), [left_int_freq_lim, right_int_freq_lim]
    
    def frequency(self):
        
        for index,row in enumerate(self.file['scan_definition'].keys()):
            tvar = self.file['scan_definition'][row][1,1]
            if tvar == b'Frequency (GHz)':
                for int_data in list(self.file['scan_definition'][row]):
                    if int_data[0] == b'start': freq_start = float(int_data[1])
                    if int_data[0] == b'stop': freq_stop = float(int_data[1])
                    if int_data[0] == b'steps': freq_steps = int(int_data[1])
        
        freq_array = np.linspace(freq_start, freq_stop, freq_steps)            
        return freq_array
        
        
    def scan_dimension(self, num=1):
        
        for index,row in enumerate(self.file['scan_definition'].keys()):
            tvar = self.file['scan_definition'][row][1,1]
            if tvar == b'ScanDimension_' + bytes(str(num), 'ascii'):
                for int_data in list(self.file['scan_definition'][row]):
                    if int_data[0] == b'steps': scan_steps = int(int_data[1])

        return scan_steps
            
        
    def choose_freq_range(self, data, int_freq_range):
        
        num_int_freq_points = data.shape[1]
        int_freq_array = np.linspace(int_freq_range[0], int_freq_range[1], num_int_freq_points)
        int_intensity = np.sum(data[:,:], axis=0)
            
        fig = plt.figure(figsize=(10,9))
        ax = fig.subplots()
            
        p, = ax.plot(int_freq_array, int_intensity)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Spectrum (counts)')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, left=0.25)
        slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider_ax_2 = plt.axes([0.1, 0.25, 0.03, 0.65])
            
        slider = RangeSlider(ax=slider_ax, 
                                label="Adjust \n frequency range", 
                                valmin=int_freq_range[0], 
                                valmax=int_freq_range[1],
                                valstep=int_freq_array)
            
        slider_2 = Slider(ax=slider_ax_2, 
                            label="Spectrum \n range", 
                            valmin=0, 
                            valmax=np.amax(int_intensity),
                            valinit=np.amax(int_intensity),
                            valfmt='%d',
                            orientation='vertical')
            
        lower_limit_line = ax.axvline(slider.val[0], color='k')
        upper_limit_line = ax.axvline(slider.val[1], color='k')
            
        def update(val):
                lower_limit_line.set_xdata([val[0], val[0]])
                upper_limit_line.set_xdata([val[1], val[1]])
                
        def update_2(val):
                ax.set_ylim(0, val)
              
        slider.on_changed(update)
        slider_2.on_changed(update_2)
        plt.show(block=True)
            
        left_index = np.where(int_freq_array == slider.val[0])[0][0]
        right_index = np.where(int_freq_array == slider.val[1])[0][0]
                
        int_rangelist = [left_index, right_index]
        int_freqlist = [slider.val[0], slider.val[1]]
        
        return int_rangelist, int_freqlist
    
    def rep_check(self):
        
        for index,row in enumerate(self.file['scan_definition'].keys()):
            tvar = self.file['scan_definition'][row][1,1]
            if tvar == b'internal - repetitions':
                for int_data in list(self.file['scan_definition'][row]):
                    if int_data[0] == b'repetitions': return int(int_data[1])
        return False