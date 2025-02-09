import os
import pickle
import numpy as np
from scipy.integrate import simpson

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,  f1_score, roc_auc_score
import matplotlib.pyplot as plt

from data_loader import load
from classifiers import simple_sens_classifier

# code to prepare classifier validation statistics for the simple sensitivity screening model for varying t.
# requires that the data is prepared and stored in "saves/sens_stat.pkl"


def prep_data( dataset=15, verify=True ): # verification was OK

    # load dataset with both fdt and qn as subject
    qn, D, y, g, all_types = load( dataset, data_column='q_n', return_depth=True )
    fdt, D_1, y_1, g_1, all_types_1 = load( dataset, data_column='f_dt', return_depth=True )

    if verify: # verify data from loads is equal
        for i, (d1, d2) in enumerate(zip(D, D_1)):
            if (d1!=d2).all():
                print(i, 'Not the same depths', d1, d2) # none found
        for i, (y1,y2) in enumerate(zip(y,y_1)):
            if (y1!=y2).all():
                print(i, 'Not the same labels', y1, y2) # none found
        for i, (g1,g2) in enumerate(zip(g,g_1)):
            if (g1!=g2).all():
                print(i, 'Not the same groups', g1, g2) # none found


    # calculate qns and std(fdt) from 30cm
    qns = []
    std_fdt = []

    # use indexing to calculate qns_30 and std std_30 from any dataset version (apart from 0.2m)
    l_0 = min(13,len(qn[0]))
    offset = int(l_0/2)
    cen_idx = int( len(qn[0])/2 ) + 1

    l_idx = cen_idx - offset - 1
    u_idx = cen_idx + offset

    for some_long_qn, some_long_fdt, some_d in zip(qn, fdt, D):
        #print( some_d )
        #print( some_d[14:27] ) # checked manually -> was as desired
        qns.append( np.average(some_long_qn[l_idx:u_idx]) ) # qns is the 0.3m windowed average
        std_fdt.append( np.std(some_long_fdt[l_idx:u_idx]) ) # same window for std(fdt)

    return qn, np.array(std_fdt), np.array(qns), y, g, all_types


# loader/saver
def load_res( f_name ):
    res = {}
    if os.path.isfile( f_name ):
        with open( f_name, 'rb') as f:
            res = pickle.load( f )
    return res
def save_res( res, f_name ):
    with open( f_name, 'wb' ) as f:
        pickle.dump( res, f )



def optimal_threshold( dataset=5 ):
    saves_file = 'saves/sens_stat.pkl'

    qn, std_fdt, qns, y, g, all_types = prep_data( dataset=dataset )

    m = np.logical_and(qns>0, std_fdt>0) # remove data with 0/neg registrations of q/std_fdt
    qn, std_fdt, qns, y, g = qn[m], std_fdt[m], qns[m], y[m], g[m]

    # Classes are : 0:quick clay, 1:Brittle, 2:Not sensitive
    # Renamed to    0:Not sensitive, 1:Sensitive

    #print(np.bincount(y)) # [1140 1700 3798] @ dataset 5
    y[y == 0] = 1 # add quick clay to sensitive index (Brittle already there)
    y[y == 2] = 0 # move Not sensitive to index 0
    prop = np.bincount(y) # [3798 2840]

    clf = simple_sens_classifier()
    clf_bound = simple_sens_classifier(apply_bounds=True)
    res = {}#load_res( saves_file )

    if dataset not in res:
        ts = np.arange(-100,201)
        #ts = np.arange(-100,201,15)
        t, acc, acc_b, precision, recall, tp, fp, tn, fn = [], [], [], [], [], [], [], [], []

        print("Calculating")
        n=len(ts)
        for i, some_t in enumerate(ts):
            if i%10==0:
                print(str(round(i/n*100,1)) + '% done', end='\r')
            clf.set_threshold( some_t )
            clf_bound.set_threshold( some_t )

            y_pred = clf.predict( qns, std_fdt )
            y_pred_b = clf_bound.predict( qns, std_fdt )
            
            t.append(some_t)
            acc.append( accuracy_score(y, y_pred) )
            acc_b.append( accuracy_score(y, y_pred_b) )
            precision.append( precision_score(y, y_pred) )
            recall.append( recall_score(y, y_pred) )

            cm = confusion_matrix( y, y_pred )
            tp.append( cm[1][1] )
            fp.append( cm[0][1] )
            tn.append( cm[0][0] )
            fn.append( cm[1][0] )

        acc = np.array(acc) * 100
        acc_b = np.array(acc_b) * 100

        idx = np.argmax(acc)
        print('Optimal threshold: t=' + str(t[idx]) + '. Accuracy: acc=' + str(np.round(acc[idx],2)) + '.' )

        res[dataset] = [ t, acc, acc_b, precision, recall, tp, fp, tn, fn ]

    else:
        print("Found saved")
  
    save_res( res, saves_file )


def plot_figure():
    ylims = [40,70]
    xlims = [-100,200]

    t, acc, acc_b, precision, recall, tp, fp, tn, fn = load_data()
    prop = [tn[0]+fp[0], tp[0]+fn[0]]

    idx = np.argmax(acc)

    # plot figure
    fig, ax = plt.subplots( figsize=(13,4), tight_layout=True)

    # asymptotes
    ax.plot( [t[0],t[-1]], [prop[0]/np.sum(prop)*100]*2, c=(.3,.3,.3), lw=1.5, ls='--' )
    ax.plot( [t[0],t[-1]], [prop[1]/np.sum(prop)*100]*2, c=(.3,.3,.3), lw=1.5, ls='--' )

    # max value
    ax.plot( [t[idx]]*2, [ylims[0],acc[idx]], c=(.3,.3,.3), lw=1.5, ls='--' )
    ax.plot( [t[idx]], [acc[idx]], marker='o', ms=6, mec=(0,0,0), mfc=(1,1,1), lw=1.5, ls='none', zorder=20 )

    ax.text( t[idx]+3, acc[idx]+0.5, '(' + str(t[idx]) + ', ' + str(round(acc[idx],1)) + '%)', fontsize=14 )

    ax.annotate(' 0 ≤ P ≤ 100', xy=(120,prop[0]/np.sum(prop)*100), xycoords='data', xytext=(105,63), textcoords='data', va='top', ha='left', fontsize=14, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.annotate('Everything classified sensitive', xy=(110,prop[1]/np.sum(prop)*100), xycoords='data', xytext=(90,50), textcoords='data', va='top', ha='left', fontsize=14, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.annotate('Nothing classified sensitive', xy=(-50,prop[0]/np.sum(prop)*100), xycoords='data', xytext=(-70,50), textcoords='data', va='top', ha='left', fontsize=14, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    # data
    ax.plot( t, acc_b,c=(237/255,28/255,46/255), lw=2, ls='--' )
    ax.plot( t, acc,c=(0,0,0), lw=2.5 )

    ax.set_xlabel( 'Threshold, ' + r'$t$' + ' (-)', fontsize=14 )
    ax.set_ylabel( 'Accuracy (%)', fontsize=14 )
    ax.tick_params( axis='both', labelsize=12 )
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


def test():
    t, acc, acc_b, precision, recall, tp, fp, tn, fn = load_data()
    idx = np.argmax(acc)

    acc_t = (tp + tn) / (tp + fp + tn + fn) * 100
    precision_t = tp / (tp+fp)
    recall_t    = tp / (tp+fn)

    if False: # tests
        print( acc_t, acc[idx] ) # OK: 67.70927772741008 67.70927772741008
        print( precision_t, precision[idx] ) # OK: 0.6655256723716382 0.6655256723716382
        print( recall_t, recall[idx] ) # OK: 0.48365316275764036 0.48365316275764036
    
    #plot_figure( t, acc, acc_b )


def load_data():
    saves_file = 'saves/sens_stat.pkl'
    res = load_res(saves_file)
    
    for r in res:
        for i in range(len( res[r] ) ):
            res[r][i] = np.array( res[r][i] )

    return res[1] # t, acc, acc_b, precision, recall, tp, fp, tn, fn



def format_axis( ax, xlim=[0,100], ylim=[0,100] ):
    fs = 16
    ax.set_xlim( xlim )
    ax.set_ylim( ylim )
    ax.xaxis.set_tick_params( labelsize=fs )
    ax.yaxis.set_tick_params( labelsize=fs )
    ax.grid()


def plot_prec_recall(f_size=(5,4)):
    fs = 16
    fs_label = 18
    dy_txt = 6
    cb=(0,142/255,194/255)

    t, acc, acc_b, precision, recall, tp, fp, tn, fn = load_data()

    precision *= 100
    recall *= 100

    id_p = np.argmax( precision[:-14])

    t_max = t[id_p]

    fig, ax = plt.subplots( figsize=f_size, tight_layout=True)
    ax.plot( recall, precision, lw=2, c=cb )



    ts = np.array([20, 40, 50, 60, 70, t_max])#np.arange(20,101,20)
    pt, rt = [], []

    for some_t in ts:
        idx = np.where(t==some_t)[0][0]
        pt = precision[idx]
        rt = recall[idx]
        ax.plot( rt, pt, marker='o', mec=(1,1,1), mew=2.5, mfc=cb, ms=8, clip_on=False, zorder=99 )

        label = str(some_t)
        if some_t==t_max: label='t = ' + label

        txt = ax.text( rt, pt+dy_txt, label, c=cb, ha='center', fontsize= fs )
        txt.set_bbox( dict(facecolor=(1,1,1), edgecolor=(1,1,1), pad=0.2))

    format_axis( ax )
    ax.set_xlabel('Recall, (%)', fontsize= fs_label)
    ax.set_ylabel('Precision, (%)', fontsize= fs_label)

    plt.savefig( 'sims_sens_prec_recall.png', dpi=150 )
    #plt.show()


def plot_roc(f_size=(5,4)):
    fs = 16
    fs_label = 18
    dy_txt = 6

    t, acc, acc_b, precision, recall, tp, fp, tn, fn = load_data()
    fpr = fp / (fp+tn)
    cg = (93/255,184/255,46/255)

    # scale
    fpr *= 100
    recall *= 100

    fig, ax = plt.subplots( figsize=f_size, tight_layout=True)
    ax.plot( fpr, recall, lw=2, c=cg, zorder = -1 )
    ax.plot( [0,100], [0,100], lw=1, ls='--', c=(.3,.3,.3),zorder = 1 )


    idx = np.argmax(recall-fpr)
    ax.plot( [fpr[idx]]*2, [fpr[idx],recall[idx]], lw=1, ls='--', c=cg,zorder = 0 )
    tx = ax.text(fpr[idx]-dy_txt*0.1, fpr[idx] + 2, 't = ' + str(t[idx]), fontsize=fs, c=cg, rotation=90, ha='left', va='bottom', rotation_mode='anchor')


    ts = np.array([30, 40, 50, 60, 80])#np.arange(20,101,20)
    pt, rt = [], []

    for some_t in ts:
        idx = np.where(t==some_t)[0][0]
        ft = fpr[idx]
        rt = recall[idx]
        ax.plot( ft, rt, marker='o', mec=(1,1,1), mew=2.5, mfc=(1,1,1), ms=8, clip_on=False, zorder=1 )
        ax.plot( ft, rt, marker='o', mec='none', mew=1, mfc=cg, ms=5, clip_on=False, zorder=10 )

        label = str(some_t)
        dx = 0
        dy = 0
        ha = 'left'
        if some_t==50: 
            label='t=' + label
            dx = -4
            dy = 7
            ha = 'right' 


        txt = ax.text( ft+dy_txt*0.3+dx, rt-dy_txt*1.1+dy, label, ha=ha, fontsize= fs )
        
        fc = (.95,.95,.95)
        if some_t==50:
            fc = (1,1,1)
        
        txt.set_bbox( dict(facecolor=fc, edgecolor=fc, pad=0.1))


    #AUC
    ax.fill_between( x=fpr, y1=recall, where= (0 < fpr)&(fpr < 100), color=(.95,.95,.95), zorder=-2 )

    auc = -simpson( recall, fpr )/10000 # fpr from high (~1) to low (~0) -> negative sign

    auc_label = 'Area under curve (AUC) = ' + str( round(auc,2) )
    txt = ax.text( 99, 2, auc_label, ha='right', fontsize= fs )
    txt.set_bbox( dict(facecolor=[.95]*3, edgecolor='none', pad=0.1))

    format_axis( ax )
    ax.set_xlabel('False positive rate, FPR (%)', fontsize= fs_label)
    ax.set_ylabel('True positive rate, TPR (%)', fontsize= fs_label)

    plt.savefig( 'sims_sens_roc.png', dpi=150 )
    #plt.show()


def plot_f1(f_size=(5,4)):
    fs = 16
    fs_label = 18
    dy_txt = 6

    t, acc, acc_b, precision, recall, tp, fp, tn, fn = load_data()
    f1 = 2/(1/precision + 1/recall)* 100

    fig, ax = plt.subplots( figsize=f_size, tight_layout=True)

    #ax.plot( t, acc, lw=2, c=(0,0,0), zorder=2, label='Accuracy' )
    ax.plot( t, f1, lw=2, c=(237/255,28/255,46/255), zorder=2, label='F' + r'$_1$' + ' score' )

    ts = np.array([30, 40, 50, 60, 80])#np.arange(20,101,20)
    pt, rt = [], []


    idx_f1 = np.argmax(f1)

    # max values
    xs = [ t[idx_f1] ]
    ys = [ f1[idx_f1] ]
    cs = [ (237/255,28/255,46/255) ]
    for x, y, c in zip(xs, ys, cs):
        ax.plot( x, y, marker='o', mec=(1,1,1), mew=2.5, mfc=c, ms=8, clip_on=False, zorder=9 )
        ax.plot( [x]*2, [0,y], lw=1, ls='--', c=c, zorder=1 )
        txt = ax.text( x, y+dy_txt*0.7, str(round(y,1))+'%', ha='center', c=c, fontsize= fs )
        txt.set_bbox( dict(facecolor=(1,1,1), edgecolor=(1,1,1), pad=0.2))

        tx = ax.text(x-dy_txt*0.1, 3, 't = ' + str(x), fontsize=fs, c=c, rotation=90, ha='left', va='bottom', rotation_mode='anchor')


    format_axis( ax, xlim=[0, 100], ylim=[0,80] )
    ax.set_xlabel('Threshold, t (-)', fontsize= fs_label)
    ax.set_ylabel('F' + r'$_1$' + ' score (%)', fontsize= fs_label)

    #ax.legend( loc='center right', fontsize=fs, framealpha=.8 )
    plt.savefig( 'sims_sens_f1.png', dpi=150 )
    #plt.show()


def plot_acc(f_size=(5,4)):
    fs = 16
    fs_label = 18
    dy_txt = 6

    t, acc, acc_b, precision, recall, tp, fp, tn, fn = load_data()
    f1 = 2/(1/precision + 1/recall)* 100

    fig, ax = plt.subplots( figsize=f_size, tight_layout=True)

    ax.plot( t, acc, lw=2, c=(0,0,0), zorder=2, label='Accuracy' )
    #ax.plot( t, f1, lw=2, c=(237/255,28/255,46/255), zorder=2, label='F' + r'$_1$' + ' score' )

    ts = np.array([30, 40, 50, 60, 80])#np.arange(20,101,20)
    pt, rt = [], []

    idx_acc = np.argmax(acc)
    #idx_f1 = np.argmax(f1)

    # max values
    xs = [ t[idx_acc] ]
    ys = [ acc[idx_acc] ]
    cs = [ (0,0,0) ]
    for x, y, c in zip(xs, ys, cs):
        ax.plot( x, y, marker='o', mec=(1,1,1), mew=2.5, mfc=c, ms=8, clip_on=False, zorder=9 )
        ax.plot( [x]*2, [0,y], lw=1, ls='--', c=c, zorder=1 )
        txt = ax.text( x, y+dy_txt*0.7, str(round(y,1))+'%', ha='center', c=c, fontsize= fs )
        txt.set_bbox( dict(facecolor=(1,1,1), edgecolor=(1,1,1), pad=0.2))

        tx = ax.text(x-dy_txt*0.1, 3, 't = ' + str(x), fontsize=fs, c=c, rotation=90, ha='left', va='bottom', rotation_mode='anchor')


    format_axis( ax, xlim=[0, 100], ylim=[0,80] )
    ax.set_xlabel('Threshold, t (-)', fontsize= fs_label)
    ax.set_ylabel('Accuracy (%)', fontsize= fs_label)

    #ax.legend( loc='center right', fontsize=fs, framealpha=.8 )
    plt.savefig( 'sims_sens_acc.png', dpi=150 )
    #plt.show()


def chart_vals():
    t, acc, acc_b, precision, recall, tp, fp, tn, fn = load_data()
    precision *= 100
    recall *= 100
    f1 = 2/(1/precision + 1/recall)

    ts = [30, 50, 70]

    res = {'acc':[],'pre':[],'rec':[],'f1':[]}

    for some_t in ts:
        idx = np.where(t==some_t)[0][0]
        res['acc'].append( acc[idx] )
        res['pre'].append( precision[idx] )
        res['rec'].append( recall[idx] )
        res['f1'].append( f1[idx] )

    [print(k,np.round(res[k],2)) for k in res.keys()]
    #print(res)
    a=1

if __name__=='__main__':
    #optimal_threshold( dataset=1 )
    #plot_figure()

    f_size = (5,4)

    print(max(0,int(0.2*21)))

    chart_vals()

    plot_prec_recall(f_size)
    plot_roc(f_size)
    plot_acc(f_size)
    plot_f1(f_size)
