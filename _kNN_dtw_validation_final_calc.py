import os
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score,  f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# own scripts
from data_loader import load
from classifiers import time_series_KNeighborsClassifier, tot_sbt


# script to run validation calculations for dtw SBT classification models.
# Because of the calculationally intensive dtw, and the number of curves in 
# the dataset, these took a very long time. ~2 days using 2 computers.
# A shorter time than the sensitivity screening study!


def validation_calc_final( res, data, ws, rs, ks, f_name, rnd_groups ):
    d_values, d_depths, d_labels, d_groups, d_class_description_dict = data # data list to variables
    all_labels = list( set(d_labels) )

    
    if rnd_groups: # randomize groups
        for i in range( np.random.randint(1000) ): np.random.shuffle(d_groups)

    cv = StratifiedGroupKFold( n_splits=10 ) # constant in study. 'squeezed almost dry': Mosteller & Tukey '68

    for some_w, some_r in zip( ws, rs ):
        print( f_name + ' - w=' + str(some_w) )

        dtw_clf = time_series_KNeighborsClassifier( n_neighbors=ks ) # <-- pass np.ndarray with k-s!
        dtw_clf.fit( d_values, all_labels, set_labels=True ) # all_labels: list of possible label values (not actual labels)
        dtw_clf.set_w( some_w )
        dtw_clf.set_endpoint_relax_r( some_r )

        last_k_id_for_run = ( ks[-1], some_w, some_r )

        if last_k_id_for_run in res:
            print('Found saved')
            continue # only calculate once


        validationa_results = [ {} for i in range(len(ks))] #        
        for ii, (tr, tt) in enumerate( cv.split(X=d_values, y=d_labels, groups=d_groups) ):
            print(str(ii+1) + '/10')

            X_train, y_train = d_values[tr], d_labels[tr]            
            X_test, y_test = d_values[tt], d_labels[tt]

            # fit and predict
            #dtw_clf.fit( qn_test, y_test ) # ONLY FOR TESTING: verify 100% train-accuracy for k=1. BIG no-no in final implementation

            dtw_clf.fit( X_train, y_train )
            y_preds_dtw = dtw_clf.predict( X_test ) # list of lists: prediction for each k
            y_probs_dtw = dtw_clf.probabilities

            for k_i, (y_pred_dtw, y_prob_dtw) in enumerate( zip(y_preds_dtw,y_probs_dtw)  ):                
                calc_fold_scores( validationa_results, k_i, y_test, y_pred_dtw, y_prob_dtw )

        for i, (some_k, some_results) in enumerate(zip(ks, validationa_results )):
            res[ ( some_k, some_w, some_r ) ] = average_results( some_results ) # combine results from all folds

        save_res( f_name, res )




def calc_fold_scores( results, k_i, y_test, y_pred, y_prob ):
    y_pred_fuzz_1 = fuzzy_match( y_test, y_pred, 1 ) # allow fuzzy - all cases
    
    metrics = [ 'acc_0', 'acc_1', 'prec_mac', 'prec_mic', 'prec_mac_1', 'prec_mic_1', 'rec_mac', 'rec_mic', 'rec_mac_1', 'rec_mic_1', 'f1_mac', 'f1_mic', 'f1_mac_1', 'f1_mic_1', 'roc_auc_ovr', 'roc_auc_ovo' ]
    for metric in metrics: # new dict: make keys with empty list
        if metric not in results[k_i]: results[k_i][metric] = []

    # store scores for each     
    results[k_i][ 'acc_0' ].append( accuracy_score(y_test, y_pred) )
    results[k_i][ 'acc_1' ].append( accuracy_score(y_test, y_pred_fuzz_1) )

    results[k_i][ 'prec_mac' ].append( precision_score(y_test, y_pred, average='macro') )
    results[k_i][ 'prec_mic' ].append( precision_score(y_test, y_pred, average='micro') ) # not relevant
    results[k_i][ 'prec_mac_1' ].append( precision_score(y_test, y_pred_fuzz_1, average='macro') )
    results[k_i][ 'prec_mic_1' ].append( precision_score(y_test, y_pred_fuzz_1, average='micro') ) # not relevant

    results[k_i][ 'rec_mac' ].append( recall_score(y_test, y_pred, average='macro') )
    results[k_i][ 'rec_mic' ].append( recall_score(y_test, y_pred, average='micro') ) # not relevant
    results[k_i][ 'rec_mac_1' ].append( recall_score(y_test, y_pred_fuzz_1, average='macro') ) # not relevant
    results[k_i][ 'rec_mic_1' ].append( recall_score(y_test, y_pred_fuzz_1, average='micro') ) # not relevant

    results[k_i][ 'f1_mac' ].append( f1_score(y_test, y_pred, average='macro') )
    results[k_i][ 'f1_mic' ].append( f1_score(y_test, y_pred, average='micro') ) # not relevant
    results[k_i][ 'f1_mac_1' ].append( f1_score(y_test, y_pred_fuzz_1, average='macro') ) # not relevant
    results[k_i][ 'f1_mic_1' ].append( f1_score(y_test, y_pred_fuzz_1, average='micro') ) # not relevant

    y_prob = prep_prob_pred( y_test, y_prob )
    
    results[k_i][ 'roc_auc_ovr' ].append( roc_auc_score( y_test, y_prob, multi_class="ovr",average="macro") ) # not relevant
    results[k_i][ 'roc_auc_ovo' ].append( roc_auc_score( y_test, y_prob, multi_class="ovo",average="macro") )


def prep_prob_pred( y_test, y_probs ):
    # Addresses imperfection in sklearn's "roc_auc_score", which only accepts probabilities.
    # for labels in y_test, and not y. This is incorrect for the validation in this study.
    # Here some test sets will not have samples from the  gravel class (with 29 elements).
    
    y_prob_labels = np.arange( len(y_probs[0]) ) # we use labels: [ 0, 1,..., n ]
    y_test_labels = np.array( list(set(y_test)) ) # unique labels in y_test
    cols_to_drop = np.setdiff1d( y_prob_labels, y_test_labels ) # find missing labels in y_test

    if len(cols_to_drop)>0: # if any
        y_probs =  np.delete( y_probs, cols_to_drop, axis=1 ) # drop missing label column(s)
        for i, row in enumerate(y_probs):
            if np.sum(row)==0: 
                y_probs[i][np.random.randint(0,len(row))]=1 # no prediction left? random alternative
        y_probs = y_probs/y_probs.sum(axis=1, keepdims=True) # rebuild probabilities ( sum=1 )

    return y_probs



def average_results( some_results ):
    for key, value in some_results.items(): # for each metric:
        some_results[ key ] = np.average( value ) # average reults from all folds

    return some_results


def fuzzy_match( y_test, y_pred, n_fuzzy ):
    { # match soil type with nearest neighbor - makes NO sense for sensitivity classes!
                           'Clay': 0,      'Silty clay': 1, 
      'Clayey silt' : 2,   'Silt': 3,      'Sandy silt': 4,
      'Silty sand'  : 5,   'Sand': 6,   'Gravelly sand': 7,
      'Sandy gravel': 8, 'Gravel': 9,
    }

    if not isinstance(y_test, np.ndarray): y_test=np.array(y_test)
    if not isinstance(y_pred, np.ndarray): y_pred=np.array(y_pred)

    y_ret = y_pred * 1

    eps = 1e-5 #y_pred floats & y_test ints, eps for floating point errors
    fuzzy_mask = np.abs( y_pred-y_test) + eps < (n_fuzzy+1)
    y_ret[fuzzy_mask] = y_test[fuzzy_mask] # within "fuzz": mark as correct
    return y_ret


def load_save( f_name ):
    if not os.path.isfile( f_name ): return {}
    with open(f_name, 'rb') as f:
        res = pickle.load( f )
    return res
def save_res( f_name, res ):
    with open(f_name, 'wb') as f:
        pickle.dump( res, f )


def calc_windows( some_d ):
    # returns np array with windows as % of sequence length
    eps = 1e-5
    dw = max( 0.04, 1/len(some_d) ) # at least 4% between steps
    return np.arange( 0, 1 + 10*eps, dw ) + eps # w as np.array


def calculate_set( set_nr, var, f_name, rnd_groups ):        
    print( 'working on dataset: ' + str(set_nr) )
    res = load_save( f_name ) # retrieve saved result dict/create new
    data = load( set_nr, var, return_depth=True ) # load desired dataset: [ qn, D, y, g, all_types ] [values, depths, classes, groups, full_group_list]

    # set algorithm parameters
    w = calc_windows( data[1][0] ) # get window range
    r = w*0 # endpoint relaxation:  w*0:no relaxation (pure dtw) / w*1: psi-dtw with r=w
    k = np.arange(1, 301, 2) # neighbor range to consider/save

    validation_calc_final( res, data, w, r, k, f_name, rnd_groups ) # incrementally saved after each w/r


def calc_sbt_scores(): # generates comparison values (purple contours)
    # load results
    f_name = os.path.join( 'saves/', 'SBT_chart.pkl' )
    res = load_save( f_name )
    
    if not res:
        # fetch data
        qn_data = load( set_nr=11, data_column='q_n', return_depth=True ) # d_values, d_depths, d_labels, d_groups, d_class_description_dict
        fdt_data = load( set_nr=11, data_column='f_dt', return_depth=True )

        # calculate desired variables
        qns = np.array( [ np.average(qn) for qn in qn_data[0] ] ) # x-axis
        std_fdt = np.array( [ np.std(f_dt) for f_dt in fdt_data[0] ] ) # y-axis
        y_true = qn_data[2] # labels
        groups = qn_data[3]

        # define & fit clf
        sbt_clf = tot_sbt() # define classifier
        sbt_clf.fit( qn_data[4] ) # fits data&clf indexes by labels ( SBT given as 1-10, where data is defined 0-9 )

        y_pred = sbt_clf.predict( qns, std_fdt )
        y_probs = gen_rnd_probs( y_pred, n_classes=10 ) # needed for fold_scores_function

        results = [ {} ]
        calc_fold_scores( results, 0, y_true, y_pred, y_probs )

    for k,v in results[0].items():
        print(k,round(v[0],4))

    print( res )


def gen_rnd_probs( y_pred, n_classes=10 ):
    y_probs = []
    for item in y_pred:
        item_probs = [ np.random.uniform() for s in range(n_classes) ] # random gibberish
        item_probs = item_probs / np.sum( item_probs ) # normalize
        y_probs.append(item_probs)
    return y_probs


def validate_dtw_and_save():
    save_basename = 'saves/' 
    
    variable = [ 'f_dt', 'q_n', 'q_ns' ][1]
    rnd_groups = False # test effects of removing work done by DBSCAN

    for set_nr in np.arange(0,10): # datasets in range 0-19
        f_name = save_basename + variable + '-dset(' + str(set_nr) + ')' + '.pkl' # where to save results
        calculate_set( set_nr, variable, f_name, rnd_groups )


if __name__=='__main__':    
    validate_dtw_and_save()
    #calc_sbt_scores()