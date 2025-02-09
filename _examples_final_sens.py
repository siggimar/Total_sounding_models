import pickle
import os
import numpy as np
from classifiers import tot_dtw_clf, tot_sbt, simple_sens_classifier, colors
from snd_file_classes.snd_files import snd_file
from snd_file_classes.method_plotter import tot_plotter
from data_loader import load # training data

import matplotlib.pyplot as plt

# Script to showcase the sensitivity screening classifies on total sounding profile.
# draws classification examples using data from SND files

example_folder = 'example_files'


def file_list( n=None ): # generates list of soundings to present
    if isinstance(n, int): n = [n]

    files = [ file for root, dirs, files in os.walk(example_folder) for file in [os.path.join(root, f) for f in files] ]
    if isinstance(n, list) or isinstance(n,np.ndarray):
         return np.array(files)[n]
    return files


def tot_data_and_figure( some_file ): # draws sounding, returns tot_plotter class instance
        # read data from file
        some_snd = snd_file( some_file )
        some_snd.parse_raw_data()        
        some_tot = some_snd.soundings['tot'][0] # files in example_folder contain a tot sounding
        some_tot.add_qn()
        some_plotter = tot_plotter( some_tot )
        some_plotter.fig_height=8
        some_plotter.plot( multiple_soundings=True )
        return some_tot, some_plotter


def classify( some_tot ): # loads classifiers, extracts desired data and returns classifications
    # load & fit GSA dtw classifier
    qn_3, y_3, _, all_types_3 = load( 3, data_column='q_n' ) # delta_D=1.0m #GSA
    dtw3_clf = tot_dtw_clf( n_neighbors=75, w=0.2 )
    dtw3_clf.fit( qn_3, y_3 )

    qn_5, y_5, _, all_types_5 = load( 5, data_column='q_n' ) # delta_D=0.5m # Sens
    dtw5_clf = tot_dtw_clf( n_neighbors=75, w=0.2 )
    dtw5_clf.fit( qn_5, y_5 )

    # define instance of simple classifers (sbt_chart, sens)
    #chart_clf = tot_sbt()
    sens_clf_4 = simple_sens_classifier(threshold=40)
    sens_clf_5 = simple_sens_classifier(threshold=50)
    sens_clf_6 = simple_sens_classifier(threshold=60)

    # get sounding data
    dtw3_data, dtw5_data, sbt_data = get_clf_data( some_tot )

    # predict dtw GSA dataset
    y_pred_dtw3 = dtw3_clf.predict( dtw3_data[0] )
    y_pred_dtw3_colors = [ colors[all_types_3[some_y]] for some_y in y_pred_dtw3 ]
    
    # predict dtw sensitivity dataset
    y_pred_dtw5 = dtw5_clf.predict( dtw5_data[0] )
    y_pred_dtw5_colors = [ colors[all_types_5[some_y]] for some_y in y_pred_dtw5 ]

    # predict simple sensitiviy model
    y_pred_sens4 = sens_clf_4.predict( sbt_data[0], sbt_data[1] )
    y_pred_sens5 = sens_clf_5.predict( sbt_data[0], sbt_data[1] )
    y_pred_sens6 = sens_clf_6.predict( sbt_data[0], sbt_data[1] )
    y_pred_sens_colors4 = sens_clf_4.class_colors( y_pred_sens4 )
    y_pred_sens_colors5 = sens_clf_5.class_colors( y_pred_sens5 )
    y_pred_sens_colors6 = sens_clf_6.class_colors( y_pred_sens6 )

    y_pred_p = sens_clf_4.P( sbt_data[0], sbt_data[1] )

    res = {         
         'DTW\n0.5m' :{ 'y_pred': y_pred_dtw3, 'd': dtw3_data[1], 'colors': y_pred_dtw3_colors },
         'DTW\n1.0m' :{ 'y_pred': y_pred_dtw5, 'd': dtw5_data[1], 'colors': y_pred_dtw5_colors },
         'Simple\nt=40' :{ 'y_pred': y_pred_sens4, 'd': sbt_data[2], 'colors': y_pred_sens_colors4 },
         'Simple\nt=50' :{ 'y_pred': y_pred_sens5, 'd': sbt_data[2], 'colors': y_pred_sens_colors5 },
         'Simple\nt=60' :{ 'y_pred': y_pred_sens6, 'd': sbt_data[2], 'colors': y_pred_sens_colors6 },
         'P' :{ 'vals': y_pred_p, 'd': sbt_data[2] },
    }

    return res


def get_clf_data( some_tot ):
    qn_05_dm, D_05_dm = get_data( some_tot, 'q_n', 0.5 )
    qn_10_dm, D_10_dm = get_data( some_tot, 'q_n', 1.0 )

    qn_03_dm, D_03_dm = get_data( some_tot, 'q_n', 0.3 )
    fd_03_dm, _ = get_data( some_tot, 'f_dt', 0.3 )

    dtw3_data = [ 
        qn_05_dm, 
        np.median(D_05_dm, axis=1) 
    ]

    dtw5_data = [ 
        qn_10_dm,
        np.median(D_10_dm, axis=1) 
    ]

    sbt_data = [
         np.average( qn_03_dm, axis=1 ),
         np.std( fd_03_dm, axis=1 ),
         np.median( D_03_dm, axis=1 )
    ]

    return dtw3_data, dtw5_data, sbt_data


def get_data( snd, param, interval ): # resamples && extracts moving windows from sounding
    eps = 1e-3
    base_interval = 0.025
    decimals = len( str(base_interval).split('.')[1] )

    d = snd.data[0] # depth
    if param == 'f_dt': x = snd.data[1] # data column
    elif param == 'q_n': x = snd.qn_data['q_n']

    # resample to 25mm depth intervals
    d_tmp = np.round(np.arange( d[0], d[-1]+eps, step=base_interval ), decimals=decimals)
    x_tmp = np.interp( d_tmp, d, x )

    # calculate the moving window size
    index_window = int( ((interval+eps) // 0.025) + 1 ) # 41 & 13 for 1.0 and 0.3
    n = len(x_tmp)-index_window + 1

    # init 2D arrays to be returned
    X = np.empty( shape=(n, index_window) )
    D = np.empty( shape=(n, index_window) )

    for i in range(n): # populate 2D arrays with moving window data
        D[i] = d_tmp[i:i+index_window]
        X[i] = x_tmp[i:i+index_window]

    return X, D


def load_example( f_name):
    vars = None
    if os.path.isfile( f_name ):
        with open(f_name, 'rb') as f:
            vars = pickle.load( f )
        print('example loaded')
    return vars


def save_example( vars, f_name ):
    with open( f_name, 'wb') as f:
        pickle.dump( vars, f )
        print('example saved')


def add_clf_to_figure( plotter, clf_res, some_tot ):
    fig = plotter.figure
    c_axs = []

    n_axs = len( clf_res.keys() )
    
    n_ax_perc = 0.6
    w_incr = (1+n_ax_perc)

    fig_width, fig_height = fig.get_size_inches()
    fig.set_size_inches( fig_width*w_incr , fig_height )

    # resize current axis
    y_lims = []

    # scale original figure axis
    right_most = 0
    left_most = 1
    first = True
    for ax in fig.axes:        
        y_lims.append(ax.get_ylim())
        ax_position = ax.get_position()
            
        if ax_position is not None:
            # Calculate the new position for the current axis
            new_left = ax_position.x0/w_incr
            new_width = ax_position.width/w_incr
            ax.set_position([new_left, ax_position.y0, new_width, ax_position.height])
            right_most = max( right_most, new_left+new_width ) # store the rightmost position
            left_most = min( left_most, new_left )

    # add the new axis
    k=-1
    ax_size = n_ax_perc/(n_axs+1)
    ax_margin = ax_size * 0.3

    # added for y-labels only
    c_axs.append( fig.add_axes( [left_most, 0.1,0.1, 0.85], zorder=-1))
    c_axs[-1].set_ylim(y_lims[-1])
    c_axs[-1].set_xticks([])

    ticks = np.arange(y_lims[-1][-1],y_lims[-1][0]+1e-3,5)
    c_axs[-1].set_yticks( ticks )
    c_axs[-1].set_yticklabels(ticks, fontsize=12)
    c_axs[-1].spines[['right', 'left', 'top', 'bottom']].set_visible(False)

    for c in clf_res:
        k+=1 
        
        ax_width = ax_size-ax_margin
        if 'y_pred' in clf_res[c]: 
            ax_width *= 0.5
        else: 
            ax_width *= 2

        c_axs.append( fig.add_axes( [right_most+ax_margin, 0.1, ax_width, 0.85] ) )
        
        c_axs[-1].set_ylim(y_lims[-1])
        c_axs[-1].set_yticks([])
        c_axs[-1].set_yticks([], minor=True)

        c_axs[-1].set_title(c)
        if 'y_pred' in clf_res[c]: 
            c_axs[-1].set_xlim(-0.5,0.5)
            #c_axs[-1].set_facecolor('xkcd:salmon')
            c_axs[-1].set_xticks([])
            c_axs[-1].set_xticks([], minor=True)

            add_classification_column( c_axs[-1], clf_res[c], some_tot )

        else: # P_plot
            c_axs[-1].set_xlim(0,100)
            c_axs[-1].plot(clf_res[c]['vals'], clf_res[c]['d'], c=(0,0,0), label='P-score')
            c_axs[-1].plot([40]*2, y_lims[-1], c=(0,142/255,194/255), ls='--', label='t=40%')
            c_axs[-1].plot([50]*2, y_lims[-1], c=(255/255,150/255,0), ls='--', label='t=50%')
            c_axs[-1].plot([60]*2, y_lims[-1], c=(237/255,28/255,46/255), ls='--', label='t=60%')
            c_axs[-1].legend()

        right_most += ax_width + ax_margin


def add_classification_column( ax, clf_res, some_tot ):

    m_alpha=0.1
    # tot_data columns:  0, 5, 6 & 7 # depth, flushing, hammering & rock drilling
    mask_ = np.logical_or.reduce([some_tot.data[5], some_tot.data[6], some_tot.data[7]]).astype(int)

    d = clf_res['d']
    mask = np.interp(d, some_tot.data[0],mask_)
    #y_s = clf_res['y_pred'] # who cares?
    c = clf_res['colors']

    inc = d[1]-d[0]
    inc_2 = inc/2

    for some_d, some_c, some_m in zip(d ,c, mask ):
        if some_m > 0.5: some_c = (some_c[0], some_c[1], some_c[2], some_c[3]*m_alpha)
        p = ax.bar('clf', inc, 1, label='', bottom=some_d-inc_2, color=some_c)
    a=1


def example_profiles( n=None, save_restore=False ):
    for i, some_file in enumerate( file_list(n) ):
        saved_ex_name = 'saves/example_' + str(i) + '.pkl'
        clf_res = None

        some_tot, plotter = tot_data_and_figure( some_file ) # cant save plotter

        if save_restore: 
            clf_res = load_example( saved_ex_name )
        
        if clf_res is None: clf_res = classify( some_tot )
        
        if save_restore:
            if not os.path.isfile( saved_ex_name ):
                save_example( clf_res, saved_ex_name )

        add_clf_to_figure( plotter, clf_res, some_tot)

        #plt.show()
        plt.savefig( 'ex_s_'+ str(n[i]) +'.png',dpi=150, transparent=False )
        a=1


if __name__=='__main__':    
    # n refers to the n-th snd file found in example_folder (not included in set)
    example_profiles( n=[1], save_restore=False )