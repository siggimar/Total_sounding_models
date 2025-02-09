import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from data_visualization_thesis import get_data, data_to_df, data_to_kde, jointplot_axs


# script used to generate thesis figures
# adapted ad-hoc to amend the presentation from proposed paper.
# done under stress of time, not my proudest work!


def get_colors():
    colors = {
        'Clay': (0, 160, 190),
        'Silty clay': (76, 188, 209),
        'Clayey silt': (154, 110, 188),
        'Silt': (112, 46, 160),
        'Sandy silt': (70, 0, 132),
        'Silty sand': (83, 181, 146),
        'Sand': (10, 150, 100),
        'Gravelly sand': (0, 119, 54),
        'Sandy gravel': (118, 118, 118),
        'Gravel': (60, 60, 60),
        'Sensitive clay': (242, 96, 108),
        'Sensitive': (242, 96, 108),
        'Quick clay': (242, 96, 108),
        'Sensitive silt': (242, 96, 108),
        'Brittle': (251, 181, 56),
        'Not sensitive': (90, 180, 50),
    }

    return { k: ( v[0]/255, v[1]/255, v[2]/255, 1 ) for k,v in colors.items() } # -> [0-1] * 4

def get_labels(detailed=False, sensitive=False):
    if sensitive:
        return [[ 'Not sensitive', 'Sensitive', ]]

    if detailed: 
        return [ ['Clay', 'Silty clay'], ['Clayey silt', 'Silt', 'Sandy silt'], ['Silty sand', 'Sand', 'Gravelly sand'], [ 'Sandy gravel', 'Gravel'] ]
    return [ 'Clay', 'Silt', 'Sand','Gravel', ]


def draw_simplest_model( ax ):
    x = np.log10( [0.1, 58.05, 9907.30, 0.1, 77.08, 9971.22] )
    y = np.log10( [0.041, 0.369, 0.0040, 1.701, 0.899, 9.969] )

    lw = 5
    ls = 'solid'
    c = ( .8, .8, .8, 0.35 )
    
    # example SBT boundaries
    ax.plot( x[0:2], y[0:2], ls=ls, lw=lw, c=c, zorder=999 )
    ax.plot( x[1:3], y[1:3], ls=ls, lw=lw, c=c, zorder=999 )    
    ax.plot( x[3:5], y[3:5], ls=ls, lw=lw, c=c, zorder=999 )
    ax.plot( x[4:6], y[4:6], ls=ls, lw=lw, c=c, zorder=999 )
    ax.plot( [x[1], x[4]], [y[1], y[4]], ls=ls, lw=lw, c=c, zorder=999 )
    ax.plot( [x[1], x[4]], [y[1], y[4]], ls=ls, lw=lw, c=c, zorder=999 )
    ax.annotate('Possible SBT boundaries\nfor primary fraction\nclassification model', xy=( np.log10(0.6), np.log10(1.6) ), xycoords='data',
        xytext=(np.log10(1.2),np.log10(12)), textcoords='data', va='center', ha='center', fontsize=16,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), bbox=dict(facecolor=(1,1,1), alpha=1, linewidth=0))

    # Haugen et al. (2016)
    ax.plot(np.log10([1000,0.01]), np.log10([10,0.0001]), lw=3, ls='solid', marker='o', ms=9, mec=(0,0,0), mew=1, mfc=(1,1,1), c=(1,1,1), zorder=998)    
    ax.annotate('Haugen et al. (2016)\nclassification line', xy=( np.log10(300), np.log10(3) ), xycoords='data',
        xytext=(np.log10(20),np.log10(45)), textcoords='data', va='center', ha='center', fontsize=16,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), bbox=dict(facecolor=(1,1,1), alpha=1, linewidth=0))


def sbt_kde_component( data_registry, varx, vary, detailed=False, d_set=1, N=100, comp_n=0, short_circuit=False ):
    is_sensitive = False#d_set<10
    
    f_size_axlabels = 18
    f_size_labelsize = 16
    
    if is_sensitive: #sensitive
        f_size_axlabels = 15
        f_size_labelsize = 13


    desired_labels = get_labels( detailed=detailed, sensitive=is_sensitive )
    desired_labels = desired_labels[ comp_n ]
    colors = get_colors()
    markers = { k: 'o' for k,v in colors.items() }

    rows, cols = 1, 1

    w_aspect_r = [6,1,0.2] * cols
    h_aspect_r = [1,6,0.2] * rows
    
    xlims = (1e-1,1e4) # haugen et al... + R
    ylims = (1e-3,1e2)
    if False:
        xlims = (1e-2,1e3)# F_DT
        ylims = (1e-3,1e2)

    #vars = ['d','f_dt', 'f_dt_lowess_20', 'f_dt_lowess_30', 'f_dt_lowess_40', 'q_n', 'q_ns', 'f_dt_res']

    x, y, labels = get_data( n=d_set, var_1=varx, var_2=vary ) #  Haugen et al: n=1: [5]&[1]
    data = data_to_df( x, y, labels, simplify=[desired_labels] )

    all_x = data['q_ns'].to_numpy()
    all_y = data['std(f_dt)'].to_numpy()
    #all_data = data_to_kde(all_x, all_y, logx=True, logy=True, N=N, xlims=xlims, ylims=ylims)

    
    fig = plt.figure( figsize=(6,5.65) )
    gs = gridspec.GridSpec( rows*3, cols*3, height_ratios=h_aspect_r, width_ratios=w_aspect_r )
    axs = []

    for i, labels in enumerate( [desired_labels] ):
        r, c = int(i/2), i%1

        if axs: # create axis
            axs.append( fig.add_subplot(gs[r*3+1, c*3], sharex=axs[0], sharey=axs[0]) ) # ax_main
        else:
            axs.append( fig.add_subplot(gs[r*3+1, c*3]) ) # ax_main

        axs.append( fig.add_subplot(gs[r*3, c*3], sharex=axs[0]) ) # ax_top            
        axs.append( fig.add_subplot(gs[r*3+1, c*3+1], sharey=(axs[0])) ) # ax_side

        # add data
        if not isinstance( labels, list ): labels = [ labels ] # to_list()
        for label in labels:
            current_data = data.loc[ data['labels']==label ]
            x = current_data['q_ns'].to_numpy()
            y = current_data['std(f_dt)'].to_numpy()
            if len(x)==0:
                print('no data for ' + label)
                continue

            if 'model_gen' not in label:
                data_registry[label] = jointplot_axs(
                    x=x, y=y,
                    label=label,
                    c=colors[label], 
                    m=markers[label],
                    i=i,
                    ax_main=axs[-3], ax_top=axs[-2], ax_side=axs[-1], 
                    xlims=xlims, ylims=ylims,
                    logx=True, logy=True, 
                    rc=(r,c), rc_lims=(rows-1,cols-1),
                    perc=50,
                    N=N,
                    add_subfig_label=False
                )
                if comp_n==0 and label==labels[-1]: axs[-3].plot(xlims, [np.log10(ylims[0]/10)]*2, lw=1.5, ls='--',c='k', label='50% bounded volume' )

        if detailed:
            if i<4: axs[-3].legend(loc='lower right', fontsize=f_size_labelsize)
            else: axs[-3].legend(bbox_to_anchor=(1.16, 1.03), loc='upper left', fontsize=f_size_axlabels)
        else:
            axs[-3].legend(loc='lower right', fontsize=f_size_axlabels)

    plt.subplots_adjust( wspace=0.00, hspace=0.00 ) # space between figures
    plt.subplots_adjust( left=0.16, right=0.99, bottom=0.1, top=0.99 )
    plt.show()
    plt.savefig('fig_' + str(d_set) +  '_sens_' + chr(ord('a')+comp_n) + '.png', dpi=150)
    
    return data


def sbt_kde_soil_density_figure(  data_registry, varx, vary, detailed=False, d_set=1, N=100, comp_n=0, SBT=True, appendix=False  ):
    is_sensitive = False#d_set<10

    desired_labels = get_labels( detailed=detailed, sensitive=is_sensitive )

    desired_labels = desired_labels[ comp_n ]
    colors = get_colors()
    markers = { k: 'o' for k,v in colors.items() }

    rows, cols = 1, 1

    w_aspect_r = [9,1,0.2] * cols
    h_aspect_r = [1,9,0.2] * rows
    if appendix:
        w_aspect_r = [6,1,0.2] * cols
        h_aspect_r = [1,6,0.2] * rows

    xlims = (1e-1,1e4) # haugen et al... + R
    ylims = (1e-3,1e2)
    if False:
        xlims = (1e-2,1e3)# F_DT
        ylims = (1e-3,1e2)

    x, y, labels = get_data( n=d_set, var_1=varx, var_2=vary ) #  Haugen et al: n=1: [5]&[1]
    data = data_to_df( x, y, labels, simplify=[desired_labels] )

    figsize = (9,8.65)
    if appendix: figsize = (6,5.65)
    fig = plt.figure( figsize=figsize )
    gs = gridspec.GridSpec( rows*3, cols*3, height_ratios=h_aspect_r, width_ratios=w_aspect_r )
    axs = []

    all_x = data['q_ns'].to_numpy()
    all_y = data['std(f_dt)'].to_numpy()
    all_data = data_to_kde(all_x, all_y, logx=True, logy=True, N=N, xlims=xlims, ylims=ylims)

    for i, labels in enumerate( [desired_labels] ):
        r, c = int(i/2), i%1

        if axs: # create axis
            axs.append( fig.add_subplot(gs[r*3+1, c*3], sharex=axs[0], sharey=axs[0]) ) # ax_main
        else:
            axs.append( fig.add_subplot(gs[r*3+1, c*3]) ) # ax_main

        axs.append( fig.add_subplot(gs[r*3, c*3], sharex=axs[0]) ) # ax_top            
        axs.append( fig.add_subplot(gs[r*3+1, c*3+1], sharey=(axs[0])) ) # ax_side

        # add data
        if not isinstance( labels, list ): labels = [ labels ] # to_list()
        
        PDFs = {} # for probability
        for label in labels:
            current_data = data.loc[ data['labels']==label ]
            x = current_data['q_ns'].to_numpy()
            y = current_data['std(f_dt)'].to_numpy()
            if len(x)==0:
                print('no data for ' + label)
                continue

            selected_model = [['Not sensitive', 'Sensitive'], [ 'Clay', 'Silt', 'Sand', 'Gravel' ]][int(SBT)]
            if detailed: selected_model = [['Clay', 'Silty clay', 'Clayey silt', 'Silt', 'Sandy silt', 'Silty sand', 'Sand', 'Gravelly sand', 'Sandy gravel', 'Gravel'], [ 'Clay', 'Silt', 'Sand', 'Gravel' ]][int(not detailed)]
            #if d_set<10: selected_model= ['Not sensitive', 'Sensitive']

            surfaces = []
            for label_s in selected_model:
                if label_s not in data_registry: continue
                surfaces.append( data_registry[label_s][2] )
                X, Y = data_registry[label_s][0], data_registry[label_s][1]
                current_data = data.loc[ data['labels']==label_s ]
                x = current_data['q_ns'].to_numpy()
                y = current_data['std(f_dt)'].to_numpy()
                jointplot_axs(
                    x=x, y=y,
                    label=label_s,
                    c=colors[label_s], 
                    m=markers[label_s],
                    i=i,
                    ax_main=axs[-3], ax_top=axs[-2], ax_side=axs[-1], 
                    xlims=xlims, ylims=ylims,
                    rc=(r,c), rc_lims=(rows-1,cols-1),
                    perc=50,
                    contours=False, fill=False, fill_dist=False, data=False, mode=True ,dist=True,
                    all_data=data_registry[label_s],
                    N=N,
                    add_subfig_label=False,
                    detailed=detailed
                )


            #probability figure
            if is_sensitive and True:
                logx=False
                logy=False
                Xn, Yn, Zn, x_n, y_n, xzn, xxn, yzn, yyn, x_topn, y_topn = list(data_to_kde( x,y, logx=logx, logy=logy, N=N, xlims=xlims, ylims=ylims ))                
                PDFs[label]= {'X': Xn, 'Y': Yn, 'Z': Zn}




            # daylight figure
            # indexed colors to use
            c_ind = [ colors[label] for label in selected_model ]
            c_ind.append((1,1,1,1))

            z_ref = np.max(all_data[3])*0.0025 # surface cutoff
            # surface stack to max indices
            stacked_surfaces = np.zeros_like( surfaces[0] )
            for surface in surfaces: stacked_surfaces = np.dstack((stacked_surfaces, surface))
            stacked_surfaces = stacked_surfaces[:,:,1:]
            stacked_surfaces = np.dstack((stacked_surfaces, np.zeros_like( surfaces[0] )+z_ref))
            max_indices = np.argmax(stacked_surfaces, axis=-1)


            # remove gravel color if gravel not in labels
            if 'Gravel' not in data_registry and colors['Gravel'] in c_ind:
                c_ind.remove(colors['Gravel'])

            # color array from max indices and indexed colors 
            colored_array = np.ones(surfaces[0].shape + (4,))
            for j in range(len(selected_model)):
                colored_array[max_indices == j] = c_ind[j]
            colored_array = colored_array.reshape(-1, 4)

            # display the image
            axs[-3].pcolormesh( X, Y, np.zeros_like(X), color=colored_array, shading='auto' )

            axs[-2].plot([-10,10], [z_ref]*2, lw=1, c='k', ls='--', zorder=20)
            axs[-1].plot([z_ref]*2, [-10,10], lw=1, c='k', ls='--', zorder=20)
            
            # dummy data for the legend
            for label in selected_model:
                if label not in data_registry: continue
                axs[-3].plot( data_registry[label][9], data_registry[label][10], marker='s', ms=8, mec=colors[label], mfc=colors[label], alpha=1, label=label, zorder=-1, ls='none' )

        if is_sensitive and True:
            # calc probability of sensitive
            eps=1e-5
            X, Y = PDFs[labels[0]]['X'], PDFs[labels[0]]['Y']
            Z0 = PDFs[labels[0]]['Z'] # Not sensitive
            Z1 = PDFs[labels[1]]['Z'] # Sensitive
            
            S = (Z1 / ( (Z0+Z1)+eps ) ) * 100
            S[max_indices==len(labels)] = np.nan

            # draw contours and label them
            CS = axs[-3].contour( X, Y, S, [5, 10,20,30,40,50,60,70,80,90], linewidths=1.5, colors=[(0,0,0,1)], linestyles='-', zorder=99, alpha=1 )
            axs[-3].clabel(CS, inline=True, fontsize=14)
        
        
        #axs[-3].legend(bbox_to_anchor=(1.16, 1.03), loc='upper center', fontsize=18)
        if not appendix:            
            pass#axs[-3].legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', fontsize=18, ncol=4)


    plt.subplots_adjust( wspace=0.00, hspace=0.00 ) # space between figures
    
    if appendix:
        axs[-3].grid( zorder=0 )
        plt.subplots_adjust( left=0.16, right=0.99, bottom=0.1, top=0.99 )
    else:
        #draw_simplest_model( axs[-3] ) # possible delineation of simplest model
        axs[-3].grid( zorder=0, c=(.9,.9,.9,.7) )
        plt.subplots_adjust( left=0.12, right=0.99, bottom=0.07, top=0.99 )

    plt.show()
    plt.savefig('fig_' + str(d_set) +  '_det_soil_density.png', dpi=150)
    


if __name__=='__main__':
    data_registry = {}

    vars = ['d','f_dt', 'f_dt_lowess_20', 'f_dt_lowess_30', 'f_dt_lowess_40', 'q_n', 'q_ns', 'f_dt_res']
    varx, vary = vars[5], vars[1] # Haugen et al:  vars[5], vars[1]

    short_circuit=False

    for d_set_i in range(10):#10, #1,2

        for comp_n in range(4):
            pass
            sbt_kde_component( data_registry, varx, vary, detailed=True, d_set=d_set_i, N=100, comp_n=comp_n, short_circuit=short_circuit )

        sbt_kde_soil_density_figure( data_registry, varx, vary, detailed=True, d_set=d_set_i, N=100, comp_n=0, SBT=True, appendix=True )
    
    