import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from data_visualization import data_to_df, data_to_kde, get_data, log_tick_formatter, set_axis_log_formatter
import matplotlib.gridspec as gridspec


# script used to generate figures in thesis, showing SBT boundaries estimated
# using a birds-eye view of KDE surfaces for point groups.
# ( most of these end up in Appendix H )


alpha = 1

model_info = {
    0:{
        'x_lim':[1e-1, 1e4],
        'y_lim':[1e-3, 1e2],
        'x_label':r'$q_{ns}$' + ' (-)',
        'y_label':r'$std(F_{DT})$' + ' (kN)',
        'var_1':'q_n',
        'var_2':'f_dt',
        'base_name': 'sens_sandard_',
    },
    1:{},
    2:{},
}



colors = {
    'Clay': ( 0, 160, 190, 1 ),
    'Silty clay': ( 76, 188, 209, 1 ),
    'Clayey silt': ( 154, 110, 188, 1 ),
    'Silt': ( 112, 46, 160, 1 ),
    'Sandy silt': ( 70, 0, 132, 1 ),
    'Silty sand': ( 83, 181, 146, 1 ),
    'Sand': ( 10, 150, 100, 1 ),
    'Gravelly sand': ( 0, 119, 54, 1 ),
    'Sandy gravel': ( 118, 118, 118, 1 ),
    'Gravel': ( 60, 60, 60, 1 ),
    'Sensitive clay': ( 242, 96, 108, 1 ),
    'Sensitive': ( 242, 96, 108, 1 ),
    'Quick clay': ( 242, 96, 108, 1 ),
    'Sensitive silt': ( 251,181,76, 1 ),
    'Brittle': ( 251, 181, 56, 1 ),
    'Not sensitive': ( 90, 180, 50, 1 ),
}

used_classes = {}

colors = { k: (v[0]/255,v[1]/255, v[2]/255, v[3] * alpha) for (k,v) in colors.items() }
r = 0.2

chart_defs = {
    'TOT simple': {
        'name': 'Simple TOT SBT chart',
        'axis': {# for axis definitions
            'logx':True, 'logy':True, 
            'xlims':(1e-1,1e4),'ylims':(1e-3,1e2)
        },
        'points': {
            'x':[0.1, 18, 62, 0.1, 56, 110, 10000, 26, 0.1, 80, 900, 10000, 10000, 0.1],  # indexed from 0
            'y':[0.035, 0.26, 0.001, 0.001, 0.35, 0.22, 0.001, 1.01, 2, 1, 3, 3, 100, 100]
        },
        'regions':{
            'Sensitive clay':[1, 2, 3, 4, 1],
            'Clay':[2, 5, 6, 7, 3, 2],
            'Sensitive silt':[1, 2, 8, 9, 1],
            'Silt':[2, 5, 10, 8, 2],
            'Sand':[5, 10, 11, 12, 7, 6, 5],
            'Gravel': [9, 8, 10, 11, 12, 13, 14, 9],
        },
        'text directions': {
            1:  ( r, -45 ), 2:  ( r, -115 ), 3:  ( r, 135 ), 4:  ( r, 35 ), 5:  ( r, 10 ), 6:  ( r, 10 ), 7:  ( r*1.8, 110 ),
            8:  ( r*1.2, 140 ), 9:  ( r, 35 ), 10: ( r*1.2, 140 ), 11: ( r*1.2, 140 ), 12: ( r*1.2, 140 ), 13: ( r*1.2, -135 ), 14: ( r*1.2, -35 )
        }
    },
    'TOT detailed': {
        'name': 'TOT SBT chart',
        'axis': {
            'x':r'$q_{ns}$' + ' (-)', 'y':r'$std(F_{DT})$' + ' (kN)',
            'logx':True, 'logy':True, 
            'xlims':(1e-1,1e4),'ylims':(1e-3,1e2)
        },
        'points': {
            'x': [0.1, 8, 9, 90, 4000, 0.1, 70, 0.1, 15, 85, 0.1, 10, 120, 340, 300, 10000, 10000, 10000, 620, 10000],
            'y': [0.035, 0.2, 0.04, 0.25, 0.001, 0.001, 0.5, 1.9, 1.15, 0.96, 100, 100, 1.2, 1.8, 0.7, 0.08, 0.001, 3, 100, 100]
        },
        'regions':{
            'Clay': [1, 2, 3, 4, 5, 6, 1],
            'Silty clay': [ 2, 7, 4, 3, 2],
            'Clayey silt': [ 9, 10, 7, 2, 9],
            'Silt': [ 8, 9, 2, 1, 8],
            'Sandy silt': [ 11, 12, 13, 10, 9, 8, 11],
            'Silty sand': [ 10, 13, 14, 15, 7, 10],
            'Sand': [ 7, 15, 16, 17, 5, 4, 7],
            'Gravelly sand': [ 14, 18, 16, 15, 14],
            'Sandy gravel': [ 19, 20, 18, 14, 19],
            'Gravel': [ 12, 19, 14, 13, 12]
        },
        'text directions': { # ( radius (log), theta (deg) )
            1:  ( r, -45 ), 2:  ( r, -115 ), 3:  ( r, -60 ), 4:  ( r, 0 ), 5:  ( r*1.5, 155 ), 6:  ( r, 35 ), 7:  ( r, -120),
            8:  ( r*1.2, 35 ), 9:  ( r, 90 ), 10: ( r*1.2, 140 ), 11: ( r*1.2, -40 ), 12: ( r*1.2, -135 ), 13: ( r*1.2, 55 ), 14: ( r*1.2, 45 ), #n
            15: ( r*1.2, 20 ), 16: ( r*1.2, -145 ), 17: ( r*1.2, 145 ), 18: ( r*1.2, 145 ), 19: ( r*1.2, -45 ),20: ( r*1.2, -135 ),
        }
    },
    'Sensitivity': {
        'name': 'TOT Sensitivity chart',
        'axis': {
            'x':r'$q_{ns}$' + ' (-)', 'y':r'$std(F_{DT})$' + ' (kN)',
            'logx':True, 'logy':True, 
            'xlims':(1e-1,1e4),'ylims':(1e-3,1e2)
        },
        'points': {'x': [], 'y': []},
        'regions':{},
        'text directions': {}
    }
}


def daylight_surface( labels, surfaces ):
    # indexed colors to use            
    c_ind = [ colors[label] for label in labels ]
    c_ind.append((1,1,1,1)) # white boundary surface

    z_ref = max( [np.max(surfaces[l]['Z']) for l in labels] ) * 0.01
    Z_ref =  surfaces[labels[0]]['Z']
    # surface stack to max indices
    stacked_surfaces = np.zeros_like( Z_ref ) # create dummy surface
    for label in labels: stacked_surfaces = np.dstack((stacked_surfaces, surfaces[label]['Z']))
    stacked_surfaces = stacked_surfaces[:,:,1:] # drop dummy surface
    stacked_surfaces = np.dstack( (stacked_surfaces, np.zeros_like( Z_ref ) + z_ref) )
    max_indices = np.argmax(stacked_surfaces, axis=-1)

    # color array from max indices and indexed colors 
    colored_array = np.ones(Z_ref.shape + (4,))
    for j in range(len(labels)):
        colored_array[max_indices == j] = c_ind[j]
    colored_array = colored_array.reshape(-1, 4)

    return colored_array, max_indices


def ax_coords( x, y, logx=True, logy=True ):
    x_ = np.log10(x) if logx else x
    y_ = np.log10(y) if logy else y

    return x_, y_


def simp_sens_coords( perc ):
    # plot very simplified % sensitive model
    theta = np.radians( 0.001*perc**2 + 0.41*perc + 71.9 )
    r1 = 0.1#0.00012*perc**2-0.02*perc+0.8
    r2 = -0.014*perc+4.8
    r3 = r2 + 0.1
    r4 = 0

    p0 = (40, 0.0006)
    p0 = tuple([np.log10(p) for p in p0])
    r = [ r1, r2, r3, r4 ]

    x = [ 10**(p0[0]+some_r*np.cos(theta)) for some_r in r ]
    y = [ 10**(p0[1]+some_r*np.sin(theta)) for some_r in r ]

    return ax_coords(x, y)


def draw_simp_sens( ax, percs, c=(0,0,0), lw=1.5 ):
    for p in percs:
        clip = not ((p==percs[0]) or (p==percs[-1]))
        
        px, py = simp_sens_coords( p )
        ax.plot( px[:2], py[:2], c=c, ls='--', clip_on=clip, lw=lw, zorder=100 )
        ax.text( px[2], py[2], str(p), color=c, fontsize=14, zorder=100, horizontalalignment='center' )
    
    ax.plot( px[3], py[3], marker='o', mec=(0,0,0), mew=1, mfc=(1,0,0), ms=8, ls='none', zorder=20, alpha=1,clip_on=False )
    ax.annotate('Pivot point for\nsimple model\n' + r'$s=\left( 40.0,\:0.0006 \right)$', xy=( px[3], py[3] ), xycoords='data', annotation_clip=False,
        xytext=(2.8,-2.2), textcoords='data', va='center', ha='center', fontsize=16,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), bbox=dict(facecolor=(1,1,1), alpha=1, linewidth=0))



def draw_sens_SBT( axs, dataset_id=0, model_id=0, N=100, chosen_model=False ):
    #model_info[model_id]['var_1']
    #model_info[model_id]['var_2']



    labels = ['Not sensitive', 'Sensitive']
    logx, logy = True, True

    ms_mode = 9
    ms_data = 7

    x, y, mat_labels = get_data(n=dataset_id, var_1='q_n', var_2='f_dt' )
    x, y = ax_coords( x, y, logx=logx, logy=logy )
    data = data_to_df( x, y, mat_labels, simplify=labels )

    eps=1e-5
    PDFs = {}
    z_ref = np.inf

    for label in labels:

        c=colors[label]

        current_data = data.loc[ data['labels']==label ]
        x = current_data['q_ns'].to_numpy()
        y = current_data['std(f_dt)'].to_numpy()
        X, Y, Z, x_, y_, xz, xx, yz, yy, x_top, y_top = all_data = list(data_to_kde( x,y, logx=False, logy=False, N=N ))

        axs[-3].plot( x_top, y_top, marker='o', mec=(0,0,0), mew=1, mfc=c, ms=ms_mode, ls='none', label=label, zorder=999, alpha=1 ) # mode
        
        # plot distributions for presented data
        axs[-1].plot( yz, yy, c=c, lw=1.5 )
        axs[-2].plot( xx, xz, c=c, lw=1.5 )

        # plot distribution modes
        axs[-1].plot( yz.max(), y_top, marker='o', mec=(0,0,0), mew=1, mfc=c, ms=ms_mode, ls='none', zorder=20, alpha=1,clip_on=False )
        axs[-2].plot( x_top, xz.max(), marker='o', mec=(0,0,0), mew=1, mfc=c, ms=ms_mode, ls='none', zorder=20, alpha=1,clip_on=False )

        PDFs[label]= {'X': X, 'Y': Y, 'Z': Z}

        used_classes[label] = -1

    color_map, max_indices = daylight_surface( labels, PDFs )
    axs[-3].pcolormesh( X, Y, np.zeros_like(X), color=color_map, shading='auto' )

    # calc probability of sensitive
    X, Y = PDFs[labels[0]]['X'], PDFs[labels[0]]['Y']
    Z0 = PDFs[labels[0]]['Z'] # Not sensitive
    Z1 = PDFs[labels[1]]['Z'] # Sensitive
    
    S = (Z1 / ( (Z0+Z1)+eps ) ) * 100
    S[max_indices==len(labels)] = np.nan

    # draw contours and label them
    CS = axs[-3].contour( X, Y, S, [5, 10,20,30,40,50,60,70,80,90], linewidths=1.5, colors=[(0,0,0,1)], linestyles='-', zorder=99, alpha=1 )
    axs[-3].clabel(CS, inline=True, fontsize=14)

    z_ref = max(np.max(Z0),np.max(Z1))*0.0025

    # cutoff values
    axs[-2].plot([-10,10], [z_ref]*2, lw=1, c='k', ls='--', zorder=21)
    axs[-1].plot([z_ref]*2, [-10,10], lw=1, c='k', ls='--', zorder=21)


    if dataset_id==1 and model_id==0: draw_simp_sens( axs[-3], np.arange(10,91,10), c=(20/255,20/255,80/255) ) # adds simple model onto density figure for Delta_d=0.3m

    axs[-3].grid( zorder=0, c=(.9,.9,.9,.7) )


def get_chart( model_id ):
    f_size_axlabels = 18
    f_size_ticklabels = 16

    rows, cols = 1, 1

    w_aspect_r = [6,1,0.2] * cols
    h_aspect_r = [1,6,0.2] * rows

    fig = plt.figure( figsize=(9,8.65) )
    #fig = plt.figure( figsize=(6,5.65) )
    gs = gridspec.GridSpec( rows*3, cols*3, height_ratios=h_aspect_r, width_ratios=w_aspect_r )
    axs = []

    r, c = 0, 0
    axs.append( fig.add_subplot(gs[r*3+1, c*3]) ) # ax_main
    axs.append( fig.add_subplot(gs[r*3, c*3], sharex=axs[0]) ) # ax_top
    axs.append( fig.add_subplot(gs[r*3+1, c*3+1], sharey=(axs[0])) ) # ax_side

    #fig, ax = plt.subplots( figsize=(9,8.65), tight_layout=True )

    logx, logy = True, True
    xlims, ylims = model_info[model_id]['x_lim'], model_info[model_id]['y_lim']


    # manual logs
    xlims_= ( np.log10(l) for l in xlims )if logx else xlims
    ylims_= ( np.log10(l) for l in ylims )if logy else ylims

    if logx: set_axis_log_formatter( axs[-3].xaxis )
    if logy: set_axis_log_formatter( axs[-3].yaxis )

    axs[-3].set_xlim( xlims_ )
    axs[-3].set_ylim( ylims_ )

    # labels and spines
    axs[-3].set_xlabel( model_info[model_id]['x_label'], fontsize=f_size_axlabels )
    axs[-3].set_ylabel( model_info[model_id]['y_label'], fontsize=f_size_axlabels )


    # turn off spines (ON: bottom+left on main + main axis on top/side)
    axs[-3].spines[ 'top' ].set_visible( False )
    axs[-3].spines[ 'right' ].set_visible( False )
    if True:
        axs[-2].spines[ 'top' ].set_visible( False )
        axs[-2].spines[ 'left' ].set_visible( False )
        axs[-2].spines[ 'right' ].set_visible( False )
        axs[-1].spines[ 'top' ].set_visible( False )
        axs[-1].spines[ 'right' ].set_visible( False )
        axs[-1].spines[ 'bottom' ].set_visible( False )

    axs[-2].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False )
    axs[-1].tick_params( axis='both', which='both', bottom=False, top=False, left=False, right=False )


    plt.setp( axs[-2].get_xticklabels(), visible=False )
    plt.setp( axs[-2].get_yticklabels(), visible=False )
    plt.setp( axs[-1].get_xticklabels(), visible=False )
    plt.setp( axs[-1].get_yticklabels(), visible=False )


    axs[-3].tick_params(axis='both', which='major', labelsize=f_size_ticklabels)
    axs[-3].minorticks_off()
    return fig, axs


if __name__=='__main__':
    var_id = 0
    for model_id in [ 0 ]:#, 1, 2 ]:
        for d_set in range(1,10):
            fig, axs = get_chart( model_id )
            draw_sens_SBT ( axs, dataset_id=d_set, model_id=model_id, N=100 )

            plt.subplots_adjust( wspace=0.00, hspace=0.00 ) # space between figures
            plt.subplots_adjust( left=0.16, right=0.99, bottom=0.1, top=0.99 )
            
            #plt.savefig(model_info[model_id]['base_name'] + str(d_set) +  '_det.png', dpi=150)            
            plt.show()
