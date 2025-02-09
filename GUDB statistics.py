import matplotlib.pyplot as plt
import numpy as np


# figure used in thesis to show trends in Norway

data = { #dilled_meters/year.  Values from a study of the available data in GUDB (except for Fredriksen et al. (1990))
    'Total sounding':
        {
            'year': [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            'm':    [154.9, 1572.9, 4958.6, 6285.6, 10816.1, 14226.0, 10019.3, 15666.9, 11468.2, 23173.2, 14597.9, 35784.2, 24727.5, 24112.9, 21328.2, 20798.8, 23027.3, 19288.6, 22862.0, 35428.5, 44621.6, 39554.7, 47495.1, 53869.1, 46015.6, 62733.0, 42154.1, 24441.4, 28559.7, 14503.3, 2178.2],
            'ls':   'solid',
            'lw':   2,
            'color': (255/255,150/255,0/255),
        },
    'Rotary pressure sounding':
        {
            'year': [1958, 1965, 1967, 1970, 1973, 1974, 1975, 1976, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
            'm':    [30.4, 4.1, 669.2, 225.0, 962.3, 30.4, 58.4, 14.8, 489.4, 771.5, 2883.7, 1041.6, 5.2, 462.1, 1110.9, 811.2, 2017.8, 620.1, 929.9, 1247.5, 1894.3, 5190.6, 1581.6, 2412.4, 6117.5, 8103.5, 10400.1, 3836.0, 2577.4, 2862.5, 2303.9, 5427.8, 2601.8, 3855.1, 7684.2, 1914.4, 3009.5, 2054.3, 1601.4, 5267.6, 1810.5, 3318.2, 5025.9, 28079.5, 11605.3, 3616.0, 1413.4, 182.8, 990.1, 54.4],
            'ls':   'solid',
            'lw':   2,
            'color': (68/255,79/255,85/255),
        },    
#    'Rotary pressure sounding, analog\ndata from Fredriksen et al. (1990)':
#        {
#            'year': [1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990],
#            'm':    [13600, 25910, 20920, 28490, 40670, 42060, 49800, 59100, 54290, 65930, 55970, 47130, 63850],
#            'ls':   'dashed',
#            'lw':   2,
#            'color': (68/255,79/255,85/255),
#        },
    'Cone penetration tests':
        {
            'year': [1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
            'm':    [203.0, 60.8, 57.5, 540.2, 480.3, 572.2, 281.2, 3405.5, 1071.1, 291.2, 743.2, 716.9, 1866.9, 158.6, 1242.2, 909.6, 1277.9, 1928.8, 2396.1, 2944.5, 3254.2, 4400.2, 3541.4, 5408.2, 2226.1, 900.6, 1820.1, 567.3, 279.2],
            'ls':   'solid',
            'lw':   2,
            'color': (237/255,28/255,46/255),
        },
}


if __name__=='__main__':

    label_size = 16
    xlim = [ 1990, 2020 ]
    #xlim = [ 1980, 2019 ]
    ylim = [ 0, 95000 ]

    fig, ax = plt.subplots(figsize=(10,6), tight_layout=True)
    
    for key, val in data.items():
        ax.plot( val['year'], val['m'], ls=val['ls'], lw=val['lw'], c=val['color'], label=key, zorder=1) #data
    
    ax.plot( data['Cone penetration tests']['year'], np.array(data['Cone penetration tests']['m'])*15, ls=(0,(1,1)), lw=2, c=(237/255,28/255,46/255), label='15 x Cone penetration tests', zorder=0) #data
    
    ax.plot( [1990], [60], ls=(0,(5,2)), lw=5, c=(93/255,184/255,46/255), zorder=0)
    
    ax.plot( [1994,1994], ylim, ls=(0,(5,2)), lw=5, c=(93/255,184/255,46/255), zorder=0)
    ax.plot( [2015,2015], ylim, ls=(0,(5,2)), lw=5, c=(93/255,184/255,46/255), zorder=0)

    # shaded area for transition period
    #x = np.arange(0,2025,1)
    #ax.fill_between( x, 0, 1, where= np.logical_and(x>=1990,x<=1995), color=(237/255,28/255,46/255), alpha=0.2, transform=ax.get_xaxis_transform(),zorder=-1)

    ax.set_xlim( xlim )
    ax.set_ylim( ylim )

    ax.tick_params(axis='both', which='major', labelsize=label_size*0.9)
    
    
    y_major = np.arange(0,90000,20000)
    ax.set_yticks(y_major)
    
    ax.set_xlabel( 'Year', fontsize=label_size*1.1 )
    ax.set_ylabel( 'Drilled meters', fontsize=label_size*1.1 )

    ax.grid(which='major', alpha=0.5)    

    ax.legend(loc='upper left', framealpha=1, fontsize=label_size)

    ax.annotate('Initial GUDB release', xy=( 2015, 85000 ), xycoords='data',
        xytext=(2006,85000), textcoords='data', va='center', ha='left', fontsize=label_size,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), bbox=dict(facecolor=(1,1,1), alpha=1.0, linewidth=0))

    ax.annotate('First NGF Total sounding \nmethod recommentadion', xy=( 1994, 60000 ), xycoords='data',
        xytext=(1998,60000), textcoords='data', va='center', ha='left', fontsize=label_size,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), bbox=dict(facecolor=(1,1,1), alpha=1.0, linewidth=0))

    #ax.annotate('Transition to digital recorders', xy=( 1995, 70000 ), xycoords='data',
    #    xytext=(1998,70000), textcoords='data', va='center', ha='left', fontsize=label_size,
    #    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), bbox=dict(facecolor=(1,1,1), alpha=1.0, linewidth=0))

    #plt.savefig( 'gudb_data.png', dpi=120 )
    plt.show()