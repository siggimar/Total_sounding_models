import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import RBFInterpolator
import os
import pickle

# script used to generate plots showing validation results from sensitivity screening models
# requires results from validataions stored in saves folder (took ~1 week to generate with two computers)

saves_folder = 'saves'

class res_plotter():
    def __init__( self, n_int_rbf=200 ):
        self.res = {}
        self.reset_fig()

        # fontsizes
        self.f_size_axlabels = 22
        self.f_size_labelsize = 20
        self.f_size_contours = 16
        self.n_int_rbf = n_int_rbf # no. rbf meshgrid inncrements

        self.chart_scores = {   # SBT chart scores on full set: 
                'acc_0': 22.39, #   used to plot contours in validation 
                'acc_1': 48.54, #   figures in thesis body
                'prec_mac': 17.21,
                'prec_mic': 22.39,
                'prec_mac_1': 38.86,
                'prec_mic_1': 48.54,
                'rec_mac': 19.91,
                'rec_mic': 22.39,
                'rec_mac_1': 43.41,
                'rec_mic_1': 48.54,
                'f1_mac': 16.39,
                'f1_mic': 22.39,
                'f1_mac_1': 37.32,
                'f1_mic_1': 48.54,
                'roc_auc_ovr': 49.25,
                'roc_auc_ovo': 49.18,
            }
        
        self.chart_scores = { # t e {30,50,70}
            'acc_0':    [ 55.98, 67.60, 64.97 ],
            'prec_mac': [ 49.08, 62.30, 75.31 ],
            'rec_mac':  [ 93.57, 60.31, 26.23 ],
            'f1_mac':   [ 64.38, 61.29, 38.90 ],
            'roc_auc_ovo': [ 0.73 ]*3,
            't':        [30,50,70]
        }


        self.chart_scores = { # t e {40,50,60}
            'acc_0':    [ 63.86, 67.60, 67.12 ],
            'prec_mac': [ 55.18, 62.30, 69.36 ],
            'rec_mac':  [ 79.89, 60.31, 40.62 ],
            'f1_mac':   [ 65.27, 61.29, 51.23 ],
            'roc_auc_ovo': [ 0.73 ]*3,
            't':        [40,50,60]
        }
        

    def set_dataset(self, n=0, w0=True, var_name='q_n' ):        
        self.w0=w0 # if r_s are set equal to w0 [ done in kNN clf and not by passing r :( ]
        loader = res_plotter.res_loader( n=n, w0=w0, var_name=var_name ) # not stored
        self.res = loader.get_data()
        self.all_metrics = self.get_metrics()


    def n_analysis( self ):
        return len( self.res.keys() )


    def reset_fig( self ):
        plt.close('all')
        self.fig, self.ax = plt.subplots( figsize=(10,5), tight_layout=True ) #(10.5,8) # #figsize=(10,4) #figsize=(10,5)#figsize=(10,4)


    def metric_settings( self, metric ):
        settings = { # store plotter settings
            'acc_0': {'label': 'Accuracy [%]', 'lims': [58,68] },
            'acc_1': {'label':'Fuzzy accuracy' + r'$_{n=1}$' + ' [%]', 'lims': [50, 64]},
            'prec_mac': {'label':'Precision, PPV [%]', 'lims': [56, 70]},
            'prec_mic': {'label':'Precision-mic, PPV [%]', 'lims': [10, 32]},
            'prec_mac_1': {'label':'Fuzzy precision, PPV [%]', 'lims': [20, 62]},
            'prec_mic_1': {'label':'Fuzzy precision_mic, PPV [%]', 'lims': [26, 64]},
            'rec_mac': {'label':'Recall, TPR [%]', 'lims': [58, 68]},
            'rec_mic': {'label':'Recall_mic, TPR [%]', 'lims': [10, 32]},
            'rec_mac_1': {'label':'Fuzzy recall, TPR [%]', 'lims': [22, 46]},
            'rec_mic_1': {'label':'Fuzzy recall_mic, TPR [%]', 'lims': [26, 64]},
            'f1_mac': {'label':r'$F_1$' + ' score, ' + r'$F_1$' + ' [%]', 'lims': [56, 66]},
            'f1_mic': {'label':r'$F_1$' + ' score_mic, ' + r'$F_1$' + ' [%]', 'lims': [10, 32]},
            'f1_mac_1': {'label':'Fuzzy ' + r'$F_1$' + ' score, ' + r'$F_1$' + ' [%]', 'lims': [18, 46]},
            'f1_mic_1': {'label':'Fuzzy ' + r'$F_1$' + ' score_mic, ' + r'$F_1$' + ' [%]', 'lims': [26, 64]},
            'roc_auc_ovr': {'label':'ROC AUC' + r'$_{total.ovr}$' +  ' [%]', 'lims': [48, 68]},
            'roc_auc_ovo': {'label':'ROC AUC' + r'$_{total.ovo}$' +  ' [%]', 'lims': [56, 74]},
        }

        return settings[ metric ]




    def get_metrics( self ):
        first_key = next(iter(self.res))
        metrics = [ m for m in self.res[first_key].keys() ]
        return metrics

    def get_ylabel( self ):
        ylabel = 'w (% length)'
        if not self.w0: ylabel = 'w, r (% length)'
        return ylabel


    def format_axis( self, metric ):
        self.ax.set_xlim( 1e-5, 300 )
        self.ax.set_ylim( 0, 100 )

        self.ax.xaxis.set_tick_params( labelsize=self.f_size_labelsize )
        self.ax.yaxis.set_tick_params( labelsize=self.f_size_labelsize )

        self.ax.set_xlabel( 'Neighbors, ' + r'$k$' + ' (-)', fontsize=self.f_size_axlabels )
        self.ax.set_ylabel( self.get_ylabel(), fontsize=self.f_size_axlabels )


    def data_from_res( self ):
        all_metrics = self.all_metrics
        scale = 100
        ks, ws, rs = [], [], []

        ms = { k: [] for k in all_metrics }

        for (k, w, r) in self.res.keys():
            tmp_r = r if self.w0 else w # wrong key stored in dict for r=w (w1)

            # store each key/value pair in dict
            ks.append( k ) # w and all metrics are in range (0-1), scale k to match
            ws.append( w*scale )
            rs.append( tmp_r*scale )

            for m in all_metrics:
                ms[m].append( self.res[ (k, w, r) ][m]*scale ) # extract each metric

        return ks, ws, rs, ms


    def get_all_zlims( self ):
        # grab data
        k, w, r, m = self.data_from_res()

        # prep results
        all_metrics = self.all_metrics
        mins = { k: np.inf for k in all_metrics }
        maxs = { k: -np.inf for k in all_metrics }

        # loop through all metrics
        for metric in all_metrics:
            mins[metric] = np.min( m[metric] )
            maxs[metric] = np.max( m[metric] )
        return mins, maxs


    def get_ranged_max( self, k_min, k_max, w_min, w_max):
        # grab data
        k, w, r, m = self.data_from_res()
        k, w, r = np.array(k), np.array(w), np.array(r)

        # prep results
        all_metrics = self.all_metrics
        maxs = { k: -np.inf for k in all_metrics }

        # consider only desired ranges
        condition = (k >= k_min) & (k <= k_max) & (w >= w_min) & (w <= w_max)
        for some_metric in m: # apply condition for each metric
            m[some_metric] = np.array(m[some_metric])
            m[some_metric] = m[some_metric][condition]


        for metric in all_metrics:
            maxs[metric] = np.max( m[metric] )

        return maxs

            

    def rbf_interp( self, x, y, z ):
        # calculate range
        xobs = np.column_stack( (x, y) )
        yobs = np.array( z )

        # calc lims
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)

        # calculate step size
        x_step = (x_max - x_min) / (self.n_int_rbf)
        y_step = (y_max - y_min) / (self.n_int_rbf)

        # interpolate over min-max grid
        xgrid = np.mgrid[x_min-2*x_step:x_max+2*x_step:x_step, y_min-2*y_step:y_max+2*y_step:y_step]
        xflat = xgrid.reshape(2, -1).T
        yflat = RBFInterpolator( xobs, yobs )( xflat )
        ygrid = yflat.reshape( xgrid[0].shape )

        return xgrid, ygrid

    def cmap( self ):
        return matplotlib.colors.LinearSegmentedColormap.from_list( "", ['blue', 'yellow', 'red'] ) #['red','orange','yellow','green','blue']

    def cl_fmt( self, x ):
        s = f'{x:.2f}'
        return rf'{s}' if plt.rcParams['text.usetex'] else f'{s}'

    def annotate_best( self, x, y, z, k_w, metric ):
        # draw point
        self.ax.scatter( x, y, s=80, fc=(1,1,1,1), ec=(0,0,0,1), zorder=10)

        # construct label
        decimals = 1
        w_label = r'$w=$' if self.w0 else r'$w=r=$'
        best_label = str( round(z,decimals) ) + '% ' + r'$k=$' + str(x) + ', ' + w_label + str( round(y,decimals) ) + '%'
        if k_w is not None:
            best_label = best_label.split('%')[0] + '%'

        # label settings
        dx, dy = 20, 20
        txt_ha = 'left'
        if x>150:
            dx *= -1 # flip sign
            txt_ha = 'right'

        if y>50:
            dy *= -1


        if x>110 and x<190:
            #dx *= 0
            txt_ha = 'center'

        t = self.ax.annotate(best_label, xy=[x,y], xytext=[x+dx,y+dy], fontsize=self.f_size_labelsize, ha=txt_ha, arrowprops=dict(arrowstyle="-", connectionstyle="arc3"), zorder=9)
        t.set_bbox(dict(facecolor=(1,1,1,1), edgecolor=(0,0,0,1)))
        


    def heatmap( self, metric='acc_0', f_name=None, k_w=None ):
        # reset figure
        self.reset_fig()
        self.format_axis( metric )

        # sets colorscale
        zlims = self.metric_settings(metric)['lims']
        norm = matplotlib.colors.Normalize(vmin=zlims[0], vmax=zlims[1])

        # get data & calc grid
        k, w, r, m = self.data_from_res()
        xgrid, ygrid = self.rbf_interp( x=k, y=w, z=m[metric] )
        
        idx_max = np.argmax( m[metric] ) if k_w is None else np.argmin((np.array(k) - k_w[0])**2 + (np.array(w) - k_w[1])**2)

        # heatmap
        p = self.ax.pcolormesh( *xgrid, ygrid, cmap=self.cmap(), norm=norm )
        cb = self.fig.colorbar( p )
                
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(self.f_size_labelsize)

        # contours
        cs = np.arange( zlims[0], zlims[1], 2 ) # select contours
        c_plt = self.ax.contour( xgrid[0], xgrid[1], ygrid, cs, linestyles=['--'], linewidths=[1], colors=[(0,0,0)] ) # draw countours
        self.ax.clabel( c_plt, c_plt.levels, inline=True, fmt=self.cl_fmt, fontsize=self.f_size_contours ) # labels


        # calculation points
        self.ax.scatter( k, w, s=1, fc=(0,0,0,0), ec=(0,0,0,0.2) ) # calc points
        self.annotate_best( k[idx_max], w[idx_max], m[metric][idx_max], k_w, metric )
        if k_w is not None:
            if metric != 'roc_auc_ovo':
                for some_c, some_t, some_score in zip([(0,1,1), (0,1,0), (0.5,0,1)], self.chart_scores['t'], self.chart_scores[metric]):
                    c_plt = self.ax.contour( xgrid[0], xgrid[1], ygrid, [ some_score ], linestyles=['-'], linewidths=[5], colors=[ some_c ] ) # draw countours
                    m_score = str( round(some_score, 1) ) + '%'
                    self.ax.plot([1,150], [-50,-50], ls='-', lw=5, c=some_c, label='t=' + str(some_t) + ': ' + m_score)
                
                plt.legend( fontsize=self.f_size_labelsize, framealpha=1, loc='upper right') #, title = "Simple model", title_fontsize=self.f_size_labelsize


        if f_name is None:
            plt.show()
        else:
            plt.savefig( f_name, dpi=100, bbox_inches='tight', pad_inches=0.05 ) #dpi=300


    class res_loader():
        def __init__( self, n=0, w0=True, var_name='q_n' ):
            # save inits
            self.n = n
            self.w0 = w0
            self.var_name = var_name

            # set folder
            self.saves_folder = saves_folder + '/w0'
            if not w0: self.saves_folder = saves_folder + '/w1'

            # load results
            self.res_file = self.get_res_file()


        def get_res_file( self ):
            d_set_str = '(' + str(self.n) + ')'
            all_files = self.get_file_list()
            res_file = [ f for f in all_files if (self.var_name in f) and (d_set_str in f)]
            return res_file[0]


        def get_file_list( self ):
            return [ f for f in os.listdir(self.saves_folder) if os.path.isfile(os.path.join(self.saves_folder, f)) ]


        def get_data( self ):
            res = {}
            file_path = os.path.join(self.saves_folder, self.res_file)
            if os.path.isfile(file_path):
                print('Accessing: ' + file_path)
                with open(file_path, 'rb') as f:
                    res = pickle.load( f )

            return res


def res_statistics():
    min_max = True # written to calibrate color scales
    max_selected_metrics = False
    max_selected_range = False



    var = ['f_dt','q_n'][1]
    w0 = [True,False][0] # this one used r=w*0
    
    plotter = res_plotter()
    plotter.set_dataset(n=10, w0=False, var_name=var)

    # prep results
    all_metrics = plotter.all_metrics
    all_metrics = [ 'acc_0', 'prec_mac', 'rec_mac', 'f1_mac', 'roc_auc_ovo' ]
    
    mins = { k: np.inf for k in all_metrics }
    maxs = { k: -np.inf for k in all_metrics }

    interesting_metrics = [ 'acc_0', 'prec_mac', 'rec_mac', 'f1_mac', 'roc_auc_ovo' ]

    max_vals = {k:[] for k in interesting_metrics}
    maxs_filtered = {k:[] for k in interesting_metrics}

    for n in range(0,10):
        plotter.set_dataset(n=n, w0=w0, var_name=var)
        tmp_mins, tmp_maxs = plotter.get_all_zlims()
        tmp_zone_max = plotter.get_ranged_max( k_min=14, k_max=36, w_min=15, w_max=35 )

        if min_max: # min-max
            for metric in all_metrics:
                mins[metric] = min( tmp_mins[metric], mins[metric])
                maxs[metric] = max( tmp_maxs[metric], maxs[metric])

        if max_selected_metrics:
            for metric in interesting_metrics:
                max_vals[metric].append( tmp_maxs[metric] )

        if max_selected_range:
            for metric in interesting_metrics:
                maxs_filtered[metric].append( tmp_zone_max[metric] )


    if min_max:
        for m in all_metrics:
            print( m + ': ' + str(round(mins[m],2)) + '-' + str(round(maxs[m],2)) )
        print('\n\n\n')

    if max_selected_metrics:
        for m in interesting_metrics:
            print( m, [round(v,2) for v in max_vals[m]] )

    if max_selected_range:
        for m in interesting_metrics:
            print( m, [round(v,2) for v in maxs_filtered[m]] )


def plot_qn_gsa( w0=False ):
    plotter = res_plotter(n_int_rbf=150)

    if False: # selecting metrics
        plotter.set_dataset(n=10, w0=False, var_name='q_n')
        metrics = plotter.all_metrics

        txt = '[' + ', \''.join(metrics) + ']'
        txt = txt.replace(',','\',')
        print(txt)

        # [ 
        #   'acc_0', 'acc_1', 'prec_mac', 'prec_mic', 
        #   'prec_mac_1', 'prec_mic_1', 'rec_mac', 'rec_mic', 
        #   'rec_mac_1', 'rec_mic_1', 'f1_mac', 'f1_mic', 'f1_mac_1', 
        #   'f1_mic_1', 'roc_auc_ovr', 'roc_auc_ovo' 
        # ]


    #chosen metrics
    metrics = [ 'acc_0', 'acc_1', 'prec_mac', 'prec_mac_1', 'rec_mac', 'rec_mac_1', 'f1_mac', 'f1_mac_1', 'roc_auc_ovo' ]
    metrics = [ 'acc_0', 'prec_mac', 'rec_mac', 'f1_mac', 'roc_auc_ovo' ]


    for var in ['q_n']: #:#['q_n','f_dt']: #
        for w0 in [False]: #[True, False]:#
            for n in range(3,4):#range(10,20): #
                for m in metrics:
                    w_0 = '_w0_' if w0 else '_w1_'
                    f_name = var + w_0 + m + '(' + str(n) + ').png'
                    f_path = os.path.join( 'figures', f_name )
                    plotter.set_dataset(n=n, w0=w0, var_name=var)
                    plotter.heatmap( metric=m, f_name=f_path, k_w=[75,20])# )#,, k_w=[21,25]



def list_all_res():
    count = 0
    for var in ['q_n','f_dt']:
        for w0 in [True, False]:
            for n in range(0,10):
                loader = res_plotter.res_loader( n=n, w0=w0, var_name=var )
                res = loader.get_data()
                count += len(res.keys())
    
    print(count)




if __name__=='__main__':
    #list_all_res()
    #res_statistics()
    plot_qn_gsa()