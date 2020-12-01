import scipy.io as sio
from numpy import *
from matplotlib import pyplot as plt 
import matplotlib
from os import listdir
from os.path import isfile, join


def main(): 
    
    # Parameters

    test_name = 'Blind' # tests available: Test1, Test2, Test3, Test4
    N = 2000 # samples per batch 
    p = 1 # number of emitting sources
 
    method = 'music' # methods available: beamf, minvar, music

    # Data acquisition & calibration

    uca = get_uca_elements_positions()
    d_m = get_data_matrix(test_name)
    c = calibration()

    # Corrected data matrix

    c_d_m    = get_corrected_data_matrix(d_m, c)
    [m, n] = shape(c_d_m)

    # azimuth, elevations angles

    el_v = zeros((int(n / N),))
    az_v = zeros((int(n / N),))

    span_az = 50
    span_el = 50

    it_az = range(-span_az, span_az + 1)
    it_el = range(-span_el, span_el + 1)
    xx, yy = meshgrid(it_az, it_el)

    #plot parameters 

    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'toolbar': 'None',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'legend.fancybox': False,
        'legend.shadow': False,
    })
    plt.ion()
    figure, ax = plt.subplots()

    for i in range(int(n / N)): 

        R = c_d_m[:, i*N:(i+1)*N ] @ matrix.getH(c_d_m[:, i*N:(i+1)*N ]) / N # correlation matrix

        if(method == "beamf"): 

            z = beamf(span_az, span_el, R, uca)            
        
        elif(method == "minvar"):

            z = minvar(span_az, span_el, R, uca)            
        
        elif(method == "music"):
            
            z = music(span_az, span_el, R, uca, p)            

        else:
            
             return -1

        ax.contour(xx, yy, z)
        plt.title(r'$P(\theta, \phi)$ contour (MUSIC method)')
        ax.set_xlabel("Azimuth $\\theta [^\circ]$")
        plt.ylabel("Elevation $\phi [^\circ]$", labelpad=-5)
        plt.grid(linestyle='dotted')
        figure.canvas.draw()
        ax.cla()

    #     el_v[i], az_v[i] = unravel_index(argmax(z, axis=None), z.shape) 
    #     az_v[i] = az_v[i] - span_az
    #     el_v[i] = el_v[i] - span_el
    #     print(i)
    # plot_shape_drawing(az_v, el_v, method)


##
#   returns calibration coefficient
##
def calibration():

    cal_data = sio.loadmat("group_c/GroupC_Test1_Cal")['Data2']
    v = zeros((8, 2000), dtype=complex_)

    for i in range(0, 16, 2): 
         v[int(i / 2),:] = cal_data[i, 0:2000] + cal_data[i+1, 0:2000] * 1j

    m = corrcoef(v)
    c = r = ones((8,), dtype=complex_); 
    
    for i in range(8):
        r[i] = m[0, i].conj()
        c[i] = 1 / r[i]; 
    
    return c


##
#   returns data matrix
##
def get_data_matrix(test_name):

    files = [f for f in listdir('group_c/') if isfile(join('group_c/', f)) and test_name + '_Mov' in f]
    data_matrix = 1; 

    for f in files:
        v = zeros((8, 2000), dtype=complex_)
        data = sio.loadmat("group_c/" + f)['Data2']

        for i in range(0, 16, 2): 
            v[int(i / 2),:] = data[i, 0:2000] + data[i+1, 0:2000] * 1j

        if isinstance(data_matrix, int): 
            data_matrix = v
        else:
            data_matrix = append(data_matrix, v, axis=1)

    return data_matrix


##
# returns corrected data matrix
##
def get_corrected_data_matrix(m, c):
    
    c_d_m = zeros(shape(m), dtype=complex_)

    for i in range(8):
        c_d_m[i, :] = c[i] * m[i, :]

    return c_d_m


##
# returns a candidate steering vector for 8-elements UCA 
##
def steering_vector(az, el, uca):

    f = 868e6
    c = 3e8

    l = c / f
    k = 2*pi / l #wave number

    sv = ones((8,1), dtype=complex_)

    for i in range(1, 8): 
        sv[i, 0] = exp(1j * k * (uca[i, 0] * cos(el * pi / 180) * sin(az * pi / 180) + uca[i, 1] * sin(el * pi / 180)))
    
    return sv


##
#   returns uca positions in array for a 8-elements UCA
##
def get_uca_elements_positions():
    r = 0.3
    a = 360 / 8
    d1 = 2 * sin( ( a / 2 )  * pi / 180) * r
    d2 = sin(a * ( 3 / 2 ) * pi / 180) * r - d1 / 2 
    uca = array([0, 0])
    uca = vstack((uca, [d2, -d2]))
    uca = vstack((uca, [d2, -(d1 + d2)]))
    uca = vstack((uca, [0, -(d1 + 2 * d2)]))
    uca = vstack((uca, [-d1, -(d1 + 2 * d2)]))
    uca = vstack((uca, [-(d1 + d2), -(d1 + d2)]))
    uca = vstack((uca, [-(d1 + d2), -d2]))
    uca = vstack((uca, [-d1, 0]))

    return uca

##
#   returns z with music algorithm
##
def beamf(span_az, span_el, R, uca):

    it_az = range(-span_az, span_az + 1)
    it_el = range(-span_el, span_el + 1)
    z = zeros((2 * span_az + 1, 2 * span_el + 1))

    for az in it_az:
        for el in it_el: 
            e = steering_vector(az, el, uca)
            eh = matrix.getH( e )
            z[ el + span_el, az + span_az] = abs((eh @ R @ e)[0, 0]) / abs((eh @ e)[0, 0])

    return z


##
#   returns z with music algorithm
##
def minvar(span_az, span_el, R, uca):

    it_az = range(-span_az, span_az + 1)
    it_el = range(-span_el, span_el + 1)
    z = zeros((2 * span_az + 1, 2 * span_el + 1))

    R_i = linalg.inv(R) 

    for az in it_az:
        for el in it_el: 
            e = steering_vector(az, el, uca)
            eh = matrix.getH( e )
            z[ el + span_el, az + span_az] = abs((eh @ R_i @ e)[0, 0])**-1
    return z


##
#   returns z with music algorithm
##
def music(span_az, span_el, R, uca, p):

    it_az = range(-span_az, span_az + 1)
    it_el = range(-span_el, span_el + 1)
    z = zeros((2 * span_az + 1, 2 * span_el + 1))

    w, v = linalg.eig(R)
    # SOURCE https://github.com/vslobody/MUSIC/blob/master/music.py
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    vn = v[:, p:len(v)]
    # END SOURCE

    for az in it_az:
        for el in it_el: 
            eh = matrix.getH( steering_vector(az, el, uca) )
            r = eh @ vn
            z[ el + span_el, az + span_az] = sum([abs(ri)**2 for ri in r])**-1
    
    return z


def plot_shape_drawing(x, y, method): 
    
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'toolbar': 'None',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'legend.fancybox': False,
        'legend.shadow': False,
        'figure.figsize': [3.9, 3.9],
    })

    fig = plt.figure('Pseudo Spectrum Estimation')
    plt.plot(x, y, 'k')
    plt.title(r'$P(\theta, \phi)$')
    plt.title('Estimated drawing shape (MUSIC method)')
    plt.xlabel('Azimuth $\\theta \ [^\circ]$')
    plt.ylabel("Elevation $\phi [^\circ]$", labelpad=-5)
    plt.grid(linestyle='dotted')

    plt.show()

if __name__ == "__main__":
    main()
