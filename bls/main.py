from bls.raw_func import BLSfile_raw
import numpy as np

class BLSfile(BLSfile_raw):
    """
    Main object for BLS data processing
    
    Parameters
    ----------
    filemask : str
        Insert the filename to process with .h5 extension
        
    Methods
    ----------
    rf_sweep(freq_range='Choose')
        BLS RF-sweep data processing
        
    linescan(length, freq_range='Choose')
        BLS linescan measurement processing
        
    map_2D(length_1, length_2, freq_range='Choose')
        BLS 2D spatial scan measurement processing
    """
    def __init__(self, filemask):
        super().__init__(filemask)
        
    def rf_sweep(self, freq_range='Choose'):
        
        """
        BLS RF-sweep data processing
        
        Parameters
        ----------
        freq_range : str or list
            What to do with frequency scan data:
            'Choose' lets you choose the frequency range to sum over + returns the frequency boundaries; 
            'All' returns raw values without summation + the full BLS scan frequency array;
            [start_freq, end_freq] returns the data summed over provided frequency range.
            
        Returns
        ----------
        BLS counts : np.array
            BLS intensity counts data, 1D, 2D or 3D depending on the freq_range value provided.
            If several repetitions of the experiment were done then the first dimension of the returned
            array indicates the number of repetitions.
            
        RF applied : np.array
            Array of frequencies applied to the antenna.
            
        Scanning frequencies : np.array or list or None
            BLS scanning frequencies depending on the value freq_range provided: 
            if 'Choose' returns list of frequency boundaries chosen;
            if 'All' returns all BLS scan frequency np.array;
            if list was provided returns None.
        """
        
        main_data, int_freq_lim = self.acquire_spectrum()
        freq_array = self.frequency()
        num_int_freq_points = main_data.shape[1]
        
        if freq_range == 'All':
            
            int_freq_array = np.linspace(int_freq_lim[0], int_freq_lim[1], num_int_freq_points)
            rep_num = self.rep_check()
            if rep_num:
                main_data = np.stack(np.split(main_data, rep_num))
            return main_data, freq_array, int_freq_array
        
        if freq_range == 'Choose':
            
            int_rangelist, int_freqlist = self.choose_freq_range(main_data, [int_freq_lim[0], int_freq_lim[1]])
            main_data = main_data[:,int_rangelist[0]:int_rangelist[1]]
            main_data = np.sum(main_data, axis=1)
            rep_num = self.rep_check()
            if rep_num:
                main_data = np.stack(np.split(main_data, rep_num))
            return main_data, freq_array, int_freqlist
        
        if isinstance(freq_range, list):
            
            int_freq_array = np.linspace(int_freq_lim[0], int_freq_lim[1], num_int_freq_points)
            left_index = np.where(int_freq_array == freq_range[0])[0][0]
            right_index = np.where(int_freq_array == freq_range[1])[0][0]
            main_data = main_data[:,left_index:right_index]
            main_data = np.sum(main_data, axis=1)
            if rep_num:
                main_data = np.stack(np.split(main_data, rep_num))
            return main_data, freq_array, None
        
    def linescan(self, length, freq_range='Choose'):
        
        """
        BLS linescan measurement processing
        
        Parameters
        ----------
        length : int
            The size of the linescan from the CCD camera.
        freq_range : str or list
            What to do with frequency scan data:
            'Choose' lets you choose the frequency range to sum over + returns the frequency boundaries; 
            'All' returns raw values without summation + the full BLS scan frequency array;
            [start_freq, end_freq] returns the data summed over provided frequency range.
            
        Returns
        ----------
        BLS counts : np.array
            BLS intensity counts data, 1D or 2D depending on the freq_range value provided
        Line coordinates : np.array
            Spatial 1D array over which the BLS scanning was performed
        Scanning frequencies : np.array or list or None
            BLS scanning frequencies depending on the value freq_range provided: 
            if 'Choose' returns list of frequency boundaries chosen;
            if 'All' returns all BLS scan frequency np.array;
            if list was provided returns None
            
        """
        main_data, int_freq_lim = self.acquire_spectrum()
        scan_steps = self.scan_dimension()
        scan_array = np.linspace(0, 1, scan_steps) * length
        num_int_freq_points = main_data.shape[1]
        
        if freq_range == 'All':
            
            int_freq_array = np.linspace(int_freq_lim[0], int_freq_lim[1], num_int_freq_points)
            return np.array(main_data).T, scan_array, int_freq_array
        
        if freq_range == 'Choose':
            
            int_rangelist, int_freqlist = self.choose_freq_range(main_data, [int_freq_lim[0], int_freq_lim[1]])
            main_data = main_data[:,int_rangelist[0]:int_rangelist[1]]
            main_data = np.sum(main_data, axis=1)
            return main_data, scan_array, int_freqlist
        
        if isinstance(freq_range, list):
            
            int_freq_array = np.linspace(int_freq_lim[0], int_freq_lim[1], num_int_freq_points)
            left_index = np.where(int_freq_array == freq_range[0])[0][0]
            right_index = np.where(int_freq_array == freq_range[1])[0][0]
            main_data = main_data[:,left_index:right_index]
            main_data = np.sum(main_data, axis=1)
            return main_data, scan_array, None
        
    def map_2D(self, length_1, length_2, freq_range='Choose'):
        
        """
        BLS 2D spatial scan measurement processing
        
        Parameters
        ----------
        length_1 : int
            The size of the first dimension of the scan area from the CCD camera.
            
        length_2 : int
            The size of the second dimension of the scan area from the CCD camera.
            
        freq_range : str or list
            What to do with frequency scan data:
            'Choose' lets you choose the frequency range to sum over + returns the frequency boundaries; 
            'All' returns raw values without summation + the full BLS scan frequency array;
            [start_freq, end_freq] returns the data summed over provided frequency range.
            
        Returns
        ----------
        BLS counts : np.array
            BLS intensity counts data, 2D or 3D depending on the freq_range value provided
        Line coordinates : np.array
            Spatial 1D array over which the BLS scanning was performed
        Scanning frequencies : np.array or list or None
            BLS scanning frequencies depending on the value freq_range provided: 
            if 'Choose' returns list of frequency boundaries chosen;
            if 'All' returns all BLS scan frequency np.array;
            if list was provided returns None
            
        """
        
        main_data, int_freq_lim = self.acquire_spectrum()
        scan_steps_1 = self.scan_dimension()
        scan_steps_2 = self.scan_dimension(num=2)
        scan_array_1 = np.linspace(0, 1, scan_steps_1) * length_1
        scan_array_2 = np.linspace(0, 1, scan_steps_2) * length_2
        num_int_freq_points = main_data.shape[1]
        
        if freq_range == 'Choose':
            
            int_rangelist, int_freqlist = self.choose_freq_range(main_data, [int_freq_lim[0], int_freq_lim[1]])
            main_data = main_data[:,int_rangelist[0]:int_rangelist[1]]
            main_data = np.sum(main_data, axis=1).reshape(scan_steps_1, scan_steps_2)
            return main_data, scan_array_1, scan_array_2, int_freqlist
        
        if isinstance(freq_range, list):
            
            int_freq_array = np.linspace(int_freq_lim[0], int_freq_lim[1], num_int_freq_points)
            left_index = np.where(int_freq_array == freq_range[0])[0][0]
            right_index = np.where(int_freq_array == freq_range[1])[0][0]
            main_data = main_data[:,left_index:right_index]
            main_data = np.sum(main_data, axis=1).reshape(scan_steps_1, scan_steps_2)
            return main_data, scan_array_1, scan_array_2
            