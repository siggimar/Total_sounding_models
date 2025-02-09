import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from data_visualization import data_to_df, data_to_kde, get_data, log_tick_formatter, set_axis_log_formatter
import matplotlib.gridspec as gridspec


# script to draw models for total soundings: SBT classification chart, sensitivity 
# screening SBT chart & simple model. # see [ if __name__=='__main__': ] block at 
# end of script

alpha = 1

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
            'x':r'$q_{ns}$' + ' (-)', 'y':r'$std(F_{DT})$' + ' (kN)',
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
    r1 = 0.00012*perc**2-0.02*perc+1.8
    r2 = -0.014*perc+4.8
    r3 = r2 + 0.1

    p0 = (40,0.0006)
    p0 = tuple([np.log10(p) for p in p0])
    r = [ r1, r2, r3 ]

    x = [ 10**(p0[0]+some_r*np.cos(theta)) for some_r in r ]
    y = [ 10**(p0[1]+some_r*np.sin(theta)) for some_r in r ]

    return ax_coords(x, y)


def draw_simp_sens( ax, percs, c=(0,0,0), lw=1.5 ):
    for p in percs:
        px, py = simp_sens_coords( p )
        ax.plot( px[:2], py[:2], c=c, ls='--', lw=lw, zorder=100 )
        ax.text( px[2], py[2], str(p), color=c, fontsize=14, zorder=100, horizontalalignment='center' )

def polygon_area( x, y ):
    a=0
    for i in range(0, len(x)-1):
        a += x[i]*y[i+1]-x[i+1]*y[i]
    return a/2

def polygon_centroid( x, y, logx=True, logy=True ): # for region label coordinates
    x_ = np.log10(x) if logx else x # account for logarithms
    y_ = np.log10(y) if logy else y

    cx, cy = 0, 0
    for i in range(0, len(x_)-1):
        cx += ( x_[i]+x_[i+1] ) * (x_[i]*y_[i+1]-x_[i+1]*y_[i])
        cy += ( y_[i]+y_[i+1] ) * (x_[i]*y_[i+1]-x_[i+1]*y_[i])

    a = polygon_area( x_, y_ )
    cx /= (6*a)
    cy /= (6*a)

    if logx: cx=np.power( 10, cx ) # account for logs
    if logy: cy=np.power( 10, cy )

    return cx, cy


def draw_sens_SBT( ax, chart_def, bg_model=None, N=200 ):
    labels = ['Not sensitive', 'Sensitive']
    logx = chart_def['axis']['logx']
    logy = chart_def['axis']['logy']

    x, y, mat_labels = get_data(n=1, var_1='q_n', var_2='f_dt' )
    x, y = ax_coords( x, y, logx=logx, logy=logy )
    data = data_to_df( x, y, mat_labels, simplify=labels )

    eps=1e-5
    PDFs = {}

    for label in labels:
        current_data = data.loc[ data['labels']==label ]
        x = current_data['q_ns'].to_numpy()
        y = current_data['std(f_dt)'].to_numpy()
        X, Y, Z, x_, y_, xz, xx, yz, yy, x_top, y_top = all_data = list(data_to_kde( x,y, logx=False, logy=False, N=N ))
        PDFs[label]= {'X': X, 'Y': Y, 'Z': Z}

        used_classes[label] = -1

    color_map, max_indices = daylight_surface( labels, PDFs )
    ax.pcolormesh( X, Y, np.zeros_like(X), color=color_map, shading='auto' )

    # calc probability of sensitive
    X, Y = PDFs[labels[0]]['X'], PDFs[labels[0]]['Y']
    Z0 = PDFs[labels[0]]['Z'] # Not sensitive
    Z1 = PDFs[labels[1]]['Z'] # Sensitive
    
    S = (Z1 / ( (Z0+Z1)+eps ) ) * 100
    S[max_indices==len(labels)] = np.nan

    # draw contours and label them
    CS = ax.contour( X, Y, S, [5, 10,20,30,40,50,60,70,80,90], linewidths=1.5, colors=[(0,0,0,1)], linestyles='-', zorder=99, alpha=1 )
    ax.clabel(CS, inline=True, fontsize=14)

    draw_simp_sens( ax, np.arange(10,91,10), c=(20/255,20/255,80/255) )



    if bg_model is not None:
        for r in bg_model['regions']:
            x_, y_ = bg_model['points']['x'], bg_model['points']['y']
            x_, y_ = ax_coords( x_, y_, logx=logx, logy=logy ) 
            rx, ry = [], []
            for p in bg_model['regions'][r]:
                rx.append( x_[p-1] )
                ry.append( y_[p-1] )

            ax.plot( rx, ry, c=(1,1,1), lw=1, zorder=2 )



def draw_manual_SBT ( ax, chart_def, simple_sens=True, extended=True ):
    x_ = chart_def['points']['x']
    y_ = chart_def['points']['y']
    logx = chart_def['axis']['logx']
    logy = chart_def['axis']['logy']
    x_, y_ = ax_coords( x_, y_, logx=logx, logy=logy ) 

    t_size = 20
    t_scale = 1.25
    t_color = ( 1, 1, 1 )
    sub_t_color = ( 1, 1, 1 )

    if extended:
        t_size = 16
        t_color = ( 0, 0, 0 )
        t_scale = 1

    for i, r in enumerate(chart_def['regions']):
        x, y = [], []
        
        # faces and boundary lines
        for p in chart_def['regions'][r]:
            x.append( x_[p-1] )
            y.append( y_[p-1] )

        vertices = [ (x_i, y_i) for (x_i,y_i) in zip(x,y) ]
        ax.add_patch( Polygon(vertices, closed=True, facecolor=colors[r], zorder=-1) )
        ax.plot( x, y, c=(0,0,0), lw=1.5, zorder=2 )

        # SBT index
        x_cen,y_cen = polygon_centroid( x, y, logx=False, logy=False )
        ax.text( x_cen, y_cen, str(i+1), verticalalignment='center', horizontalalignment='center', color=t_color, size=t_size, zorder=200)
        
        used_classes[r] = i+1
    
    # region vertices
    ax.plot(x_, y_, ls='none', marker='o', ms=7, mec='k', mew=1.5, mfc=(1,1,1),zorder=10, clip_on=False, )
    for i, (px, py) in enumerate(zip(x_,y_)):
        s = chart_def['text directions'][i+1]

        tx = px + s[0]*np.cos(s[1]*np.pi/180)
        ty = py + s[0]*np.sin(s[1]*np.pi/180)

        ax.text(tx,ty,chr(ord('a') + i), color=sub_t_color, verticalalignment='center', horizontalalignment='center', size=12*t_scale, zorder=200)
    
    if simple_sens: draw_simp_sens( ax, np.arange(50,91,40), c=(1,1,1) ) # chart too busy with this



def SBT_chart( chart_def ):
    f_size_axlabels = 18
    f_size_ticklabels = 16
    fig, ax = plt.subplots( figsize=(9,8.65), tight_layout=True )

    logx, logy = chart_def['axis']['logx'], chart_def['axis']['logy']
    
    xlims = chart_def['axis']['xlims']
    ylims = chart_def['axis']['ylims']

    # manual logs
    xlims_= ( np.log10(l) for l in xlims )if logx else xlims
    ylims_= ( np.log10(l) for l in ylims )if logy else ylims

    if logx: set_axis_log_formatter( ax.xaxis )
    if logy: set_axis_log_formatter( ax.yaxis )

    ax.set_xlim( xlims_ )
    ax.set_ylim( ylims_ )

    # labels and spines
    ax.set_xlabel( chart_def['axis']['x'], fontsize=f_size_axlabels )
    ax.set_ylabel( chart_def['axis']['y'], fontsize=f_size_axlabels )
    ax.spines[ 'right' ].set_visible( False )
    ax.spines[ 'top' ].set_visible( False )
    ax.tick_params(axis='both', which='major', labelsize=f_size_ticklabels)
    ax.minorticks_off()

    return fig, ax


def two_SBT_charts( chart_defs ):
    f_size_axlabels = 14
    f_size_ticklabels = 12

    h_aspect_r, width_ratios = [ 5, 1 ], [4, 4, 1.25]

    axs = []
    fig = plt.figure( figsize=(12,7) )
    gs = gridspec.GridSpec(2, 3, height_ratios=h_aspect_r, width_ratios=width_ratios )
    
    axs.append( fig.add_subplot(gs[0, 0]) )
    axs.append( fig.add_subplot(gs[0, 1]) )
    axs.append( fig.add_subplot(gs[0, 2]) )
    
    bottom_subplot_spec = gs[1, :].subgridspec(1, 1, wspace=0.0)
    #ax4 = fig.add_subplot(bottom_subplot_spec[0, 0])
    axs.append( fig.add_subplot(gs[1, :]) )
    l_adjust = 0
    axs[-1].set_position(
        [
            axs[-1].get_position().bounds[0] + l_adjust, 
            axs[-1].get_position().bounds[1], 
            axs[-1].get_position().bounds[2], 
            axs[-1].get_position().bounds[3]
        ]
    )

    #fig, axs = plt.subplots( 1,3, figsize=(12,6), gridspec_kw={'width_ratios': [4, 4, 1.25]})#, tight_layout=True )

    # format axis
    for i, (ax, chart_def) in enumerate(zip( axs, chart_defs )):
        logx, logy = chart_def['axis']['logx'], chart_def['axis']['logy']
        
        xlims = chart_def['axis']['xlims']
        ylims = chart_def['axis']['ylims']

        # manual logs
        xlims_= ( np.log10(l) for l in xlims )if logx else xlims
        ylims_= ( np.log10(l) for l in ylims )if logy else ylims

        if logx: set_axis_log_formatter( ax.xaxis )
        if logy: set_axis_log_formatter( ax.yaxis )

        ax.set_xlim( xlims_ )
        ax.set_ylim( ylims_ )

        # labels and spines
        ax.set_xlabel( chart_def['axis']['x'], fontsize=f_size_axlabels )
        ax.set_ylabel( chart_def['axis']['y'], fontsize=f_size_axlabels )
        ax.spines[ 'right' ].set_visible( False )
        ax.spines[ 'top' ].set_visible( False )
        ax.tick_params(axis='both', which='major', labelsize=f_size_ticklabels)
        ax.minorticks_off()

    axs[2].axis('off')
    
    axs[3].axis('off')
        
    return fig, axs

def point_defs( ax, chart_def ):
    c = (.0,.0,.0)
    pts = chart_def['points']
    dx, dy = .9, 1

    for i, (px, py) in enumerate( zip(pts['x'], pts['y']) ):
        j=i+1
        y_max = j*dy
        ax.text( 0 * dx, j*dy , chr(ord('a') + i), color=c, verticalalignment='center', horizontalalignment='center', size=11, zorder=200) #, weight='bold'
        ax.text( 2 * dx, j*dy , str(py), color=c, verticalalignment='center', horizontalalignment='center', size=10, zorder=200)
        ax.text( 1 * dx, j*dy , str(px), color=c, verticalalignment='center', horizontalalignment='center', size=10, zorder=200)
    
    for i in range(3): 
        ax.text( i * dx, -0.7 , ['Point\n(#)', r'$q_{ns}$'+'\n(-)', r'$std(F_{DT})$'+'\n(kN)' ][i], color=c,verticalalignment='center', horizontalalignment='center', size=11, zorder=200)

    ax.set_ylim((-1.5,y_max+1))
    ax.set_xlim((.8,3*dx))

    ax.invert_yaxis()


def SBT_legend( chart_def ):
    f_size_zones = 16
    f_size_txt = 15
    fig, ax = plt.subplots( figsize=(9,2), tight_layout=True )

    
    dx, cx, dy, cy = 0.35, 2.0, 0.5, -0.8

    rows = 3

    for i, region in enumerate(chart_def['regions']):
        j=i+1
        r, c = j % rows, j//rows

        vertices = [ (c*cx, r*cy), (c*cx+dx, r*cy), (c*cx+dx, r*cy+dy), (c*cx, r*cy+dy), (c*cx, r*cy) ]
        ax.add_patch( Polygon(vertices, closed=True, facecolor=colors[region], zorder=-1, edgecolor=(0,0,0), lw=1.5) )
        ax.text( c*cx+dx/2, r*cy+dy/2, str(j), c=(1,1,1), verticalalignment='center', horizontalalignment='center', size=f_size_zones, zorder=200)
        ax.text( c*cx + 1.3 * dx, r*cy+dy/2, region, verticalalignment='center', horizontalalignment='left', size=f_size_txt, zorder=200)
        t = str(j) if j>-1 else ''

    c = int((j)/rows)
    ax.plot( [10],[0] )
    ax.set_xlim(-0.2,8)
    ax.axis('off')

    return fig, ax

def legend( ax ):
    dx, cx, dy, cy = 0.5, 3.0, 0.5, 1

    rows = 2
    k = -1

    for key, value in used_classes.items():
        k += 1
        r, c = k%rows, int(k/rows)

        vertices = [ (c*cx, r), (c*cx+dx, r), (c*cx+dx, r+dy), (c*cx, r+dy), (c*cx, r) ]
        ax.add_patch( Polygon(vertices, closed=True, facecolor=colors[key], zorder=-1, edgecolor=(0,0,0), lw=1.5) )
        if value !=-1: ax.text( c*cx+dx/2, r+dy/2, str(value), verticalalignment='center', horizontalalignment='center', size=11, zorder=200)
        ax.text( c*cx + 1.3 * dx, r+dy/2, key, verticalalignment='center', horizontalalignment='left', size=11, zorder=200)
        t = str(value) if value>-1 else ''

    c = int((k+1)/rows)
    ax.plot( [c*cx, c*cx + 2*dx], [0+dy/2, 0+dy/2], ls='-', lw=1.5, c=(0,0,0))    
    ax.plot( [c*cx, c*cx + 2*dx], [1+dy/2, 1+dy/2], ls='--', lw=1.5, c=(0,0,0))
    t = ax.text( c*cx + dx, 0+dy/2, '60', verticalalignment='center', horizontalalignment='center', size=10, zorder=200)
    t.set_bbox(dict(facecolor=(1,1,1), alpha=1, edgecolor=(1,1,1)))

    ax.text( c*cx + 2.5*dx, 0+dy/2, '% Sensitive', verticalalignment='center', horizontalalignment='left', size=11, zorder=200)
    ax.text( c*cx + 2.5*dx, 1+dy/2, '% Simplified', verticalalignment='center', horizontalalignment='left', size=11, zorder=200)

    ax.set_xlim((-dx/2,22))    
    ax.set_ylim((1.7, -0.3))
    #ax.axis('equal')
    ax.xaxis.set_visible(False)    
    ax.yaxis.set_visible(False)


if __name__=='__main__':
    if False: # tot-classification chart
        fig, ax = SBT_chart(chart_defs['TOT detailed'] ) #= two_SBT_charts( (chart_defs['TOT detailed'], chart_defs['Sensitivity']) )
        draw_manual_SBT ( ax, chart_defs['TOT detailed'], simple_sens=False, extended=False ) # draws everything on left ax
        plt.savefig( 'Simple_SBT.png',dpi=150, transparent=False )
        plt.show()        
        
    if True: # detailed legend
        fig, ax = SBT_legend( chart_defs['TOT detailed'] )
        plt.savefig( 'Simple_SBT_legend.png',dpi=150, transparent=False )
        plt.show()
        

    if True: # sensitivity screening model attempt at drawing right axis
        fig, ax = SBT_chart(chart_defs['TOT detailed'] )
        draw_sens_SBT ( ax, chart_defs['Sensitivity'], chart_defs['TOT detailed'],N=200 )
        plt.show()


    if False:
        fig, axs = two_SBT_charts( (chart_defs['TOT detailed'], chart_defs['Sensitivity']) )
        draw_manual_SBT ( axs[0], chart_defs['TOT detailed'] ) # draws everything on left ax
        draw_sens_SBT ( axs[1], chart_defs['Sensitivity'], chart_defs['TOT detailed'] ) # draws everything on right ax
        point_defs( axs[2], chart_defs['TOT detailed'] ) # table with point coordinates to right of figure
        legend( axs[3] )


        margins = 0.08
        margins_b = margins/3
        plt.subplots_adjust(left=margins, bottom=margins_b, right=1, top=1-margins_b, wspace=0.3, hspace=0.3)

        tx, ty = -0.16, 1.03
        axs[0].text(tx, ty, 'A', verticalalignment='top', horizontalalignment='left', transform=axs[0].transAxes, fontsize=20)
        axs[1].text(tx, ty, 'B', verticalalignment='top', horizontalalignment='left', transform=axs[1].transAxes, fontsize=20)

        plt.savefig( 'Simple_SBT.png',dpi=600, transparent=False )
        plt.show()