import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
import pickle
from dtw_naive import dtw_cls


# script used to show that although the triangle inequality is not 
# guaranteed for the dtw-distance, the number of cases where it fails
# is low,  especially as the sequences get longer.
#
# this figure did not make it into the thesis

def triangle_inequality(  early_out=False, from_file='' ):
    from dtaidistance import dtw as dtw_c

    def thousand_separator(x, pos):
        return '{:,.0f}'.format(x).replace(',', ' ')

    colors = {
        0:(237,28,46), # red
        1:(0,142,194), # blue
        2:(93,184,46), # green
        3:(255,150,0), # orange
        4:(68,79,85), # dark gray
        5:(112,48,160), # purple
    }
    colors = { k:(r/255,g/255,b/255) for k, (r,g,b) in colors.items() } # rgb to r%, g%, b%

    np.random.seed = 1234 # for repeatability

    solutions = {}
    sums = {}
    early_out_n = 20000

    N_iter = int(1e13)
    n_to_test = 9
    l = 0
    u = 9

    plotting ={}

    if from_file !='' and os.path.isfile( from_file ):
        with open(from_file, 'rb') as f:
            plotting, tested, sums, solutions = pickle.load( f )
    else:
        for n in range(3, n_to_test):
            solutions = {}
            sums = {}

            x = []
            y = []

            tested = {}
            for i in range(N_iter):
                a = [ np.random.uniform(l,u) for i in range(n) ]
                b = [ np.random.uniform(l,u) for i in range(n) ]
                c = [ np.random.uniform(l,u) for i in range(n) ]

                some_id = tuple(a+b+c)

                if some_id in tested: continue
                if not a: continue

                tested[some_id]=some_id

                if False: # naïve
                    ac = dtw_cls(a,c, p=2).DTW_distance
                    ab = dtw_cls(a,b, p=2).DTW_distance
                    bc = dtw_cls(b,c, p=2).DTW_distance
                else: # c++ implementation (faster - but w. p=2)
                    a = np.array(a).astype(np.double)
                    b = np.array(b).astype(np.double)
                    c = np.array(c).astype(np.double)
                    ac = dtw_c.distance( a, c )
                    ab = dtw_c.distance_fast( a, b )
                    bc = dtw_c.distance_fast( b, c )

                val = ac <= ab+bc

                if not val:
                    some_id = hash( tuple(a+b+c) )
                    if id not in sums:
                        sums[some_id] = sum( a + b + c )
                        solutions[some_id] = [ a, b, c ]

                        x.append(len(tested))
                        y.append(len(solutions.keys()))

                if len(tested) % 1000 == 0:
                    print( 'n = ' + str(n) + '.  Tested ' + str(len(tested)) + '. found: ' + str(len( solutions.keys() )) + '. (i:' + str(i) + ')', end='\r' )
                    if len(tested) % 100000 == 0:
                        print( 'n = ' + str(n) + '.  Tested ' + str(len(tested)) + '. found: ' + str(len( solutions.keys() )) + '. (i:' + str(i) + ')' )
                if len(tested)==1000000 or (early_out and len(tested)>=early_out_n): break 
            plotting[n] = (x.copy(), y.copy())
        
        if from_file !='':
            with open(from_file, 'wb') as f:
                pickle.dump( [ plotting, tested, sums, solutions ], f )

        #if sums: # some elements in sum
        #    break

        print('\n')
    
    fig = plt.figure( figsize=(10,5) )
    ax = fig.add_axes((0.10, 0.12, 0.77, 0.86))

    #fig, ax = plt.subplots(figsize=(8,6))
    for i, n in enumerate(plotting):
        t_val = '%.2f' % round((plotting[n][1][-1]/plotting[n][0][-1])*100,2)
        ax.plot( plotting[n][0], plotting[n][1], c=colors[i], label='n = ' + str( n ), lw=2 )
        ax.text( len(tested) * 1.01, plotting[n][1][-1], 'n=' + str(n) + ': ' + str(t_val) + '%', verticalalignment='center', size=14 )

    ax.yaxis.set_major_formatter( FuncFormatter(thousand_separator) )
    ax.xaxis.set_major_formatter( FuncFormatter(thousand_separator) )

    plt.legend( title='Sequence length', title_fontsize=14, fontsize=12 )
    ax.set_xlabel('Unique combinations (-)', fontsize=16)
    ax.set_ylabel('(9) does not apply (-)', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_ticks(np.arange(0, len(tested)+1, step=int(len(tested)/5)))
    ax.set_ylim(bottom=0)
    ax.set_xlim(0,len(tested))
    ax.spines[['right', 'top']].set_visible(False)
    #ax.ticklabel_format( style='plain' )
    plt.show()

    # find/report smalles
    key = min(sums, key=sums.get)
    min_val = sums[key]

    smallest = [ v for k, v in solutions.items() if sums[k]==min_val ]

    print( 'smallest solutions with sum_length=' + str(sums[key]))
    for s in smallest:
        print( s )
        print( str(dtw_cls(s[0],s[2]).DTW_distance) + '>' + str(dtw_cls(s[0],s[1]).DTW_distance) + '+' + str(dtw_cls(s[1],s[2]).DTW_distance) ) 

def verfy_triangle_inequality_simple():
    '''
        Visual check of the candidates
        
    '''
    n=3 # example to use

    M = [ # all examples of failed metric with length=3 and binary values
        [[0, 1, 1], [0, 0, 1], [0, 0, 0]], # 2>0+1
        [[1, 1, 0], [1, 0, 0], [0, 0, 0]], # 2>0+1
        [[0, 0, 0], [0, 0, 1], [0, 1, 1]], # 2>1+0
        [[0, 0, 0], [1, 0, 0], [1, 1, 0]], # 2>1+0
    ]

    a = M[n][0] 
    b = M[n][1]
    c = M[n][2] 

    if False:
        ans = dtw_cls(a,c) # 2/2/2/2
    elif False:
        ans = dtw_cls(a,b) # 0/0/1/1
    else:
        ans = dtw_cls(b,c) # 1/1/0/0

    ans.plot()
    # hence dtw_cls(a,c)>dtw_cls(a,b)+dtw_cls(b,c) := 2>1+0/2>0+1 for the provided examples -> triangle inequality fails


if __name__=='__main__':
    triangle_inequality( early_out=False, from_file='dtw_triangle_ineq.pkl' )
    #verfy_triangle_inequality_simple()