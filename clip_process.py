import os
import numpy as np
from scipy.fft import fft, fft2
from scipy.signal.windows import hamming
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import matplotlib.pyplot as plt

subjectID = 6
#file_name = r'normal_pieces_'+ str(subjectID)
#file_name = r'yawn_pieces_'+ str(subjectID)
#file_name = r'nod_pieces_'+ str(subjectID)
file_name = r'blink_pieces_'+ str(subjectID)

radar_path = r'D:\gy\fatigueDetection\data_3'
radarDatafile = os.path.join(radar_path,file_name+'.mat')
#file_name_output = r'normal_pieces_val'
#file_name_output = r'yawn_pieces_val'
#file_name_output = r'nod_pieces_val'
file_name_output = r'blink_pieces_val'

mat_data = loadmat(radarDatafile)
signal_pieces = mat_data['signal_pieces']
Freqs = mat_data['Freqs']

n_RX = 20
n_TX = 20
fft2d_dim = 30
Smat = []
# print(mat_data)
print(signal_pieces.shape)
_,clips = signal_pieces.shape
print(signal_pieces.dtype)
print(Freqs.shape)

for i in range(0,clips):
    print(i)
    Smat = signal_pieces[0,i]
    print(Smat.shape)
    frame_length, ant, range_length = Smat.shape
    
    Nfft_range = int(2**(np.ceil(np.log2(Smat.shape[2])) + 1))
    print(Nfft_range)
    
    temp_radar_cube = np.zeros((frame_length, fft2d_dim, fft2d_dim, Nfft_range), dtype=np.complex128)
    
    
    for frame_ind in range(frame_length):
        if frame_length<=3:
            break
        Smat_frame = Smat[frame_ind, :, :]

        range_win = hamming(range_length)
        ant_win = hamming(n_TX)
        frame_complex = np.reshape(Smat_frame, [n_RX, n_TX, range_length])

        temp_fft_1 = np.zeros((n_RX, n_TX, range_length), dtype=np.complex128)
        temp_fft_2 = np.zeros((n_RX, n_TX, range_length), dtype=np.complex128)
        temp_fft = np.zeros((n_RX, n_TX, range_length), dtype=np.complex128)

        for ant_rx in range(n_RX):
            for ant_tx in range(n_TX):
                temp_fft_1[ant_rx, ant_tx, :] = frame_complex[ant_rx, ant_tx, :] * range_win

        for ant_rx in range(n_RX):
            for range_bin in range(range_length):
                temp_fft_2[ant_rx, :, range_bin] = temp_fft_1[ant_rx, :, range_bin] * ant_win

        for ant_rx in range(n_TX):
            for range_bin in range(range_length):
                temp_fft[ant_rx, :, range_bin] = temp_fft_2[ant_rx, :, range_bin] * ant_win

        temp_fft_range = np.zeros((n_RX, n_TX, Nfft_range), dtype=np.complex128)
        for ant_rx in range(n_RX):
            for ant_tx in range(n_TX):
                temp = fft(temp_fft[ant_rx, ant_tx, :], Nfft_range)
                temp_fft_range[ant_rx, ant_tx, :] = temp
        #MTI_spectrum_1 = np.abs(temp_fft_range[:, :, range_bin] - 0.99 * temp_fft_range[:, :, range_bin - 1])

        temp_fft_angle = np.zeros((fft2d_dim, fft2d_dim, Nfft_range), dtype=np.complex128)
        
        for range_bin in range(Nfft_range):
            temp_fft_angle[:, :, range_bin] = fft2(temp_fft_range[:, :, range_bin], (fft2d_dim, fft2d_dim))
            temp_radar_cube[frame_ind,:,:,range_bin] = np.abs(temp_fft_angle[:, :, range_bin])
            if frame_ind != 0:                
                MTI_spectrum_2 = np.abs(temp_fft_angle[:, :, range_bin] - 0.99 * temp_fft_angle[:, :, range_bin - 1])
                guassSpectrum_2 = gaussian_filter(MTI_spectrum_2, sigma=1)
        interest_radar_cube = temp_radar_cube[:,:,:,180:220]
        with open(f'{file_name_output}_{i}.npy', 'wb') as f:
            np.save(f, interest_radar_cube)

