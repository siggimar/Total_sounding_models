'''
This script contains a naïve dtw implementation to showcase the inner workings of the algorithm.

Endpoint constraint relaxation is added, but windowing omitted.

Two visualization functions are available:  
    dtw_cls.plot_alignment()    -   shows both sequences before and after warp
    dtw_cls.plot()              -   shows the distance matrix and warp path



Running this module produces simple example 
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
import numpy as np




class dtw_cls():
    def __init__( self, x, y, r=0, omega=0, p=2 ):
        self.std_val = np.inf
        self.dx, self.dy = 0.5, 0.5

        self.d = self.get_dist( kernel='minowski')
        self.p = p

        self.accent_i, self.accent_j = self.std_val, self.std_val

        self.x, self.m = x, len( x ) # sequences
        self.y, self.n = y, len( y )        
        self.r = r # relaxation factor
        self.omega = omega # penalty

        # accumulative distance matrix        
        self.D = []
        for i in range(self.m + 1):
            row = []
            for j in range(self.n + 1):
                row.append( dtw_cls.square( self, i, j ))
            self.D.append( row )

        self.align()
        self.generate_path()


    def get_path( self ):
        return self.path


    def generate_path( self ):
        # path to first endpoint stored in squares instances 
        self.path = []
        current_element = self.first_endpoint
        while current_element.parent is not None: # backtrack
            self.path.append( (current_element.i, current_element.j) )
            current_element = current_element.parent
            if len(self.path)==1 and current_element.boundary: 
                self.path = []
                self.first_endpoint = current_element

        self.last_endpoint = current_element

        self.path.reverse() # prefer left to right


    def align( self ):
        for i in range(self.r+1): # apply starting boundary condition
            self.D[0][i].set_value( None, 0 )
            self.D[i][0].set_value( None, 0 )
            
            self.D[i][0].boundary = True
            self.D[0][i].boundary = True
            self.D[self.m-i][self.n].boundary = True
            self.D[self.m][self.n-i].boundary = True

        for i in range(1,self.m+1): # calculate D(i,j)
            for j in range(1, self.n+1):
                d = self.d( self.x[i-1], self.y[j-1], self.p ) # sequence starting idx=0 !

                D_vals = {}
                D_vals[ (i-1,j-1) ] = self.D[i-1][j-1].value          # match
                D_vals[ (i-1,j) ] = self.D[i-1][j].value + self.omega # insertion                
                D_vals[ (i,j-1) ] = self.D[i][j-1].value + self.omega # deletion

                key = min(D_vals, key=D_vals.get) # possibly implement rule for case:equal
                self.D[i][j].set_value( self.D[key[0]][key[1]], D_vals[key]+d )

        ends = {} # generate path
        for i in range( self.r+1): 
            ends[(self.m-i,self.n)] = self.D[self.m-i][self.n].value
            ends[(self.m,self.n-i)] = self.D[self.m][self.n-i].value
        key = min(ends, key=ends.get)

        self.DTW_distance = ( ends[key] )**(1/self.p)

        self.first_endpoint = self.D[key[0]][key[1]]


    def warped_y( self, full=False ):
        # construct new warped sequence
        first_offset = self.path[0][1]-self.path[0][0]
        last_offset  = self.path[-1][1]-self.path[-1][0]

        warped_j = [p[0] for p in self.path]
        warped_y = [self.y[p[1]-1] for p in self.path]

        if not full: return warped_y, warped_j

        pref_j = []
        pref_y = []
        for i in range(0,self.path[0][1]-1):
            pref_j.append( self.path[0][0] - first_offset + i )
            pref_y.append( self.y[i] )

        suff_j = []
        suff_y = []
        for i in range( self.path[-1][1], len(self.y) ):
            suff_j.append( i-last_offset + 1 )
            suff_y.append( self.y[i] )

        return pref_y + warped_y + suff_y , pref_j + warped_j + suff_j


    def get_dist( self, kernel='minowski' ):
        '''Minkowski Distance implemented, with standard value of p=2'''
        def minkowski( a, b, p ):
            return (np.abs(a-b))**p
        def squared( a, b, p=2 ):
            return (a-b)**p
        def linear( a, b ):
            return abs(a-b)


        if kernel=='minowski': return minkowski
        if kernel=='squared': return squared
        return linear


    def showcase( self, i, j ):
        self.accent_i = i
        self.accent_j = j

        self.D[i][j].set_accent( soft=False )
        self.D[i-1][j-1].set_accent( soft=True )
        self.D[i-1][j].set_accent( soft=True )
        self.D[i][j-1].set_accent( soft=True )


    def plot( self ):
        # plots D matrix, sequences, boundaries and alignment path 
        self.fig, self.ax = plt.subplots( 2,2, figsize=(11,6), gridspec_kw={'width_ratios': [1, 8], 'height_ratios': [6, 1]}, tight_layout=True)

        xlims = ( -2-self.dx, self.m+self.dx*1.07 )
        ylims = ( -2-self.dy, self.n+self.dy*1.07 )

        for i in range( 0, self.m+1 ):
            for j in range( 0, self.n+1 ):
                if self.D[i][j].value!=np.inf:
                    value = '{:.0f}'.format(self.D[i][j].value)
                    f_size = 10
                else:
                    value = r'$\infty$'
                    f_size = 13

                weight = 'normal'
                if self.D[i][j].accent and not self.D[i][j].soft_accent:
                    weight = 'bold'
                # values and boundaries
                self.ax[0][1].add_patch( self.D[i][j].draw() )
                self.ax[0][1].text( i, j, value, weight=weight, verticalalignment='center', horizontalalignment='center', fontsize=f_size, zorder=10 )

        # text values and indexes
        self.ax[0][1].text( 0, -1, 'x', verticalalignment='center', horizontalalignment='center', fontsize=f_size, c=(0/255,142/255,194/255), zorder=10 )
        self.ax[0][1].text( -1, -2, 'i', fontstyle='italic', verticalalignment='center', horizontalalignment='center', fontsize=f_size, c=(.3,.3,.3), zorder=10 )
        for i in range( 0, self.m+1 ):
            c=(0,0,0) if i==self.accent_i else (0/255,142/255,194/255)
            weight='bold' if i==self.accent_i else 'normal'
            if i>0: self.ax[0][1].text( i, -1, '{:.0f}'.format(self.x[i-1]), weight=weight, verticalalignment='center', horizontalalignment='center', fontsize=f_size, c=c, zorder=10 )
            self.ax[0][1].text( i, -2, i, fontstyle='italic', verticalalignment='center', horizontalalignment='center', fontsize=f_size, c=(.3,.3,.3), zorder=10 )
        self.ax[0][1].text( -1, 0, 'y', verticalalignment='center', horizontalalignment='center', fontsize=f_size, c=(237/255,28/255,46/255), zorder=10 )
        self.ax[0][1].text( -2, -1, 'j', verticalalignment='center', horizontalalignment='center', fontsize=f_size, c=(.3,.3,.3), zorder=10 )
        for j in range( 0, self.n+1 ):
            c=(0,0,0) if j==self.accent_j else (237/255,28/255,46/255)
            weight='bold' if j==self.accent_j else 'normal'
            if j>0: self.ax[0][1].text( -1, j, '{:.0f}'.format(self.y[j-1]), weight=weight, verticalalignment='center', horizontalalignment='center', fontsize=f_size, c=c, zorder=10 )
            self.ax[0][1].text( -2, j, j, fontstyle='italic', verticalalignment='center', horizontalalignment='center', fontsize=f_size, c=(.3,.3,.3), zorder=10 )

        # highlight accented indexes
        if self.accent_i<len(self.x)+1 and self.accent_j<len(self.y)+1:
            self.ax[0][1].add_patch( Rectangle( (self.accent_i-self.dx, -1-self.dy), 1, 1, facecolor = (1,1,0), fill=True, zorder=0) )
            self.ax[0][1].add_patch( Rectangle( (-1-self.dx, self.accent_j-self.dy), 1, 1, facecolor = (1,1,0), fill=True, zorder=0) )
            x1 = self.accent_i-self.dx - 1
            y1 = self.accent_j-self.dx - 1

            x_vals = [ x1, x1+2, x1+2, x1, x1 ]
            y_vals = [ y1+2, y1+2, y1, y1, y1+2 ]

            self.ax[0][1].plot( x_vals, y_vals, c=(0,0,0), lw=1.1, zorder=15 )

        # draw sequences
        self.ax[1][1].plot( np.arange(1,len(self.x)+1), self.x, marker='o', ms=5, mec=(0,0,0), mew=0.8, c=(0/255,142/255,194/255) )
        self.ax[0][0].plot( self.y, np.arange(1,len(self.y)+1), marker='o', ms=5, mec=(0,0,0), mew=0.8, c=(237/255,28/255,46/255) )

        # draw path
        x_path = [elm[0] for elm in self.path ]
        y_path = [elm[1] for elm in self.path ]
        self.ax[0][1].plot( x_path, y_path, lw=6, marker='o', ms=16,ls='-', c=(93/255,184/255,46/255,1), zorder=4 )
        self.ax[1][1].plot( [p[0] for p in self.path], [min( self.x )]*len(self.path), marker='o', ms=3, lw=1, ls='-', c=(93/255,184/255,46/255,1), zorder=1 )
        self.ax[0][0].plot( [min( self.y )]*len(self.path), [p[1] for p in self.path], marker='o', ms=3, lw=1, ls='-', c=(93/255,184/255,46/255,1), zorder=1 )


        # set all axis limits
        self.ax[0][1].set_xlim( xlims )
        self.ax[0][1].set_ylim( ylims )
        self.ax[0][0].set_ylim( ylims )
        self.ax[1][1].set_xlim( xlims )
        self.ax[0][0].invert_xaxis()
        self.ax[1][1].invert_yaxis()

        # turn all axis off
        self.ax[0][0].axis('off')
        self.ax[0][1].axis('off')
        self.ax[1][0].axis('off')
        self.ax[1][1].axis('off')
        plt.show()


    def plot_warp( self ):
        self.fig = plt.figure( figsize=(10,4), tight_layout=True )
        gs = self.fig.add_gridspec( 2, hspace=0.5 )
        self.ax = gs.subplots(sharex=True, sharey=True)

        #self.fig, self.ax = plt.subplots( 2, 1, figsize=(10,4), gridspec_kw={'height_ratios': [1, 1]}, tight_layout=True )

        # plot original sequences
        self.ax[0].plot( np.arange(1,len(self.x)+1), self.x, marker='o', ms=5, mec=(0,0,0), mew=0.8, c=(0/255,142/255,194/255), label='x', zorder=3 )
        self.ax[0].plot( np.arange(1,len(self.y)+1), self.y, marker='o', ms=5, mec=(0,0,0), mew=0.8, c=(237/255,28/255,46/255), label='y', zorder=3 )

        for p in self.path:
            xy_a = [ p[0], self.x[p[0]-1] ]
            xy_b = [ p[1], self.y[p[1]-1] ]
            con = ConnectionPatch(xyA=xy_a, xyB=xy_b, coordsA="data", coordsB="data",
                      axesA=self.ax[1], axesB=self.ax[0], color=(93/255,184/255,46/255,1), lw=.5, ls='-', zorder=-1) # (.6,.6,.6)
            self.ax[1].add_artist(con)


        warped_y, warped_j = self.warped_y()
        warped_y_full, warped_j_full = self.warped_y( full=True )

        used_i, used_x = [], []
        for i in np.arange(1,len(self.x)+1):
            for p in self.path:
                if i==p[0]:
                    used_i.append( i )
                    used_x.append( self.x[i-1] )
                    break


        self.ax[1].plot( np.arange(1,len(self.x)+1), self.x, marker='o', ms=5, mec=(.6,.6,.6), mew=0.8, c=(.6,.6,.6), zorder=1 )
        self.ax[1].plot( used_i, used_x, marker='o', ms=5, mec=(0,0,0), mew=0.8, c=(0/255,142/255,194/255), label='x', zorder=5 )
        
        self.ax[1].plot( warped_j, warped_y, marker='o', ms=5, mec=(0,0,0), mew=0.8, c=(237/255,28/255,46/255), label='y - warped', zorder=5)
        self.ax[1].plot( warped_j_full, warped_y_full, marker='o', ms=5, mec=(.6,.6,.6), mew=0.8, c=(.6,.6,.6), label='not considered', zorder=1)


        x_lims = [np.inf, -np.inf]
        for i in range(len(self.ax)):
            tmp_lims = self.ax[i].get_xlim()
            x_lims[0] = min(x_lims[0], tmp_lims[0])
            x_lims[1] = max(x_lims[1], tmp_lims[1])


        for i in range(len(self.ax)):
            self.ax[i].set_xlim(x_lims)
            self.ax[i].set_ylim(bottom=0)
            self.ax[i].spines[['right', 'top']].set_visible(False)
        self.ax[0].plot([x_lims[1]+100]*2, [min(self.x)]*2, c=(93/255,184/255,46/255,1), ls='-', lw=.5, label='warp path') #(.6,.6,.6)

        self.ax[0].legend(loc='lower right')
        self.ax[1].legend(loc='lower right')

        self.ax[1].set_xlabel('indexes, i and j (-)', loc='right')
        self.ax[0].set_ylabel('sequence values (-)', loc='top')
        self.ax[1].set_ylabel('sequence values (-)', loc='top')

        self.ax[1].xaxis.set_ticks(np.arange(-2,26))

        plt.tight_layout()
        plt.show()
        exit()




    class square():
        def __init__( self, model, i, j ):
            self.model = model
            self.parent = None
            self.i, self.j = i, j
            self.value = self.model.std_val
            self.boundary = False
            self.accent = False
            self.soft_accent = False


        def set_value( self, parent, value ):
            self.parent = parent
            self.value = value


        def set_accent( self, soft=False ):
            self.accent=True
            self.soft_accent=soft


        def draw( self ):
            facecolor = 'None'
            if self.boundary:
                facecolor = (.9,.9,.9)
            if self.accent:
                facecolor = (1,1,0)#(255/255,150/255,0/255, 0.2)

            return Rectangle( (self.i-self.model.dx, self.j-self.model.dy), 1, 1,
                    edgecolor = (0.5,0.5,0.5),
                    facecolor = facecolor,
                    fill=True,
                    lw=0.5,
                    zorder=2
                )


def showcase_example( plot_warp=True, reverse=False):
    if not reverse:
        x = [ -0.9, 2.7, 4.4, 4.1, 2.1, 2.3, 2.2, 3.2, 3.1, 3.3, 2.6, 2.6, 2.6, 1.8, 1.4, 1.5, 1.8, 2.0, 1.7, 1.9, 2.5, 2.6, 2.7, 2.8, 2.9]
        y = [  -0.1, -0.2, -0.3, -0.4, 2.6, 3.8, 3.3, 1.4, 1.8, 2.5, 2.8, 2.7, 2.1, 1.2, 1.3, 1.1, 1.3, 2.2 ]
    else:
        y = [ -0.9, 2.7, 4.4, 4.1, 2.1, 2.3, 2.2, 3.2, 3.1, 3.3, 2.6, 2.6, 2.6, 1.8, 1.4, 1.5, 1.8, 2.0, 1.7, 1.9, 2.5, 2.6, 2.7, 2.8, 2.9]
        x = [  -0.1, -0.2, -0.3, -0.4, 2.6, 3.8, 3.3, 1.4, 1.8, 2.5, 2.8, 2.7, 2.1, 1.2, 1.3, 1.1, 1.3, 2.2 ]

    x = np.array( x ) * 10 + 45 # get rid of decimals
    y = np.array( y ) * 10 + 45 

    dtw = dtw_cls( x, y, r=3, omega=0, p=1 )

    if plot_warp: dtw.plot_warp()
    dtw.showcase( i=13, j=7)
    dtw.plot()


if __name__=='__main__':
    showcase_example( plot_warp=True, reverse=False )