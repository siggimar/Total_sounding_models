import matplotlib.pyplot as plt

from curve_correlation import cptu_rate_dataset
from curve_correlation import depth_adjuster
from curve_correlation import cptu_units

'''
Module to showcase correlation of tip-resistance curves.

5 curves from 
   *  NGTS: Tiller-Flotten quick clay site in Trondheim
   *  NTNU: Halsen silt site in Stjørdal
are shown in separate diagrams.

'''

def get_curves():    
    d_sets = []
    unit_models = []
    d_adjusters = []

    for location in [ 0, 2 ]:
        save_name = str( location ) + '.pkl'
        
        d_sets.append( cptu_rate_dataset.dataset() )
        
        d_sets[-1].load_dataset( location=location, from_file=save_name, read_logfiles=True )
        d_sets[-1].save_to_file( save_name ) # saves tons of time 2nd time around
    
        unit_models.append( cptu_units.unit_model(d_sets[-1]) )

        d_adjusters.append( depth_adjuster.depth_adjuster(d_sets[-1], unit_models[-1]) )
        d_adjusters[-1].set_warps()

    return d_sets

def plot_correlation_data( datasets ):
    location_f_size = 18
    title_f_size = 14
    tick_f_size = 12
    fig, axs = plt.subplots( 1, len(datasets), figsize=(10,5), tight_layout=True )
    
    
    for ax in axs:
        ax.set_ylabel('Depth (m)', fontsize=title_f_size )
        ax.set_xlabel( 'Corrected tip resistance, ' + r'$q_t$' + ' (kPa)', fontsize=title_f_size )
        
    
    
    for i, dataset in enumerate( datasets ):
        #axs[i].set_title( dataset.location_name, fontsize=location_f_size )
        for s in dataset.soundings:
            d, data = s.get_data_with_detph( 'qt', warped_depth=True, apply_drops=False )
            axs[i].plot( data, d, label=s.pos_name)


    axs[0].set_xlim(0, 3e3)
    axs[1].set_xlim(0, 3e3)
    axs[0].set_ylim(11,7)
    axs[1].set_ylim(11,7)
    #axs[0].invert_yaxis()

    for ax in axs:
        ax.xaxis.set_label_position('top')
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=tick_f_size)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.legend( fontsize=tick_f_size )
    fig.text(0.01, 0.94, 'A', fontsize=location_f_size, weight='bold', color=(0,0,0), verticalalignment='center', horizontalalignment='left')
    fig.text(0.51, 0.94, 'B', fontsize=location_f_size, weight='bold', color=(0,0,0), verticalalignment='center', horizontalalignment='left')


    plt.show()


if __name__=='__main__':
    datas = get_curves()
    plot_correlation_data( datas )

    a=1