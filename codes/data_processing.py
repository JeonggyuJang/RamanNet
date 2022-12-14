"""
    Helper function for data procssing
"""

import numpy as np 

def segment_spectrum(spectrum, w=50, dw=25):
    """
    Segment the raman spectrum into overlapping windows

    Args:
        spectrum (numpy array): input raman spectrum
        w (int, optional): length of window. Defaults to 50.
        dw (int, optional): step size. Defaults to 25.

    Returns:
        numpy array: array of segmented raman spectrum
    """

    return np.array([spectrum[i:i+w] for i in range(0,len(spectrum)-w,dw) ])

    ### inefficient ###
    #segments = []
    #for i in range(0,len(sig)-w,dw):
    #    segments.append(sig[i:i+w])

    #return np.array(segments)

def segment_spectrum_batch(spectra_mat, w=50, dw=25):
    """
    Segment multiple raman spectra into overlapping windows

    Args:
        spectra_mat (2D numpy array): array of input raman spectrum
        w (int, optional): length of window. Defaults to 50.
        dw (int, optional): step size. Defaults to 25.

    Returns:
        list of numpy array: list containing arrays of segmented raman spectrum
    """
    
    return [spectra_mat[:,i:i+w] for i in range(0,spectra_mat.shape[1]-w,dw) ]
    '''
    temp = [spectra_mat[:,i:i+w] for i in range(0,spectra_mat.shape[1]-w,dw) ]
    tt = np.array(temp).swapaxes(0,1)
    t2=tt[:]
    t3=t2.reshape(-1,t2.shape[1],50)
    t3 = list(t3)
    t4=[]
    for i in range(t2.shape[0]):
        t4.append(np.array(t3[i]))
    print(np.array(t4).shape)
    return t4'''
