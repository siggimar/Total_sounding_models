import os
import json
from data_loader import load # training data

# script to generate a tremmed version of the representative dataset (keep points in group 201)

def trim_reduced():
    f_name = os.path.join('training_data','d_set_rko_999m_reduced.json')
    f_name_r = os.path.join('training_data','d_set_rko_9999m_reduced.json')
    with open( f_name, 'r' ) as f:
        data = json.load( f )
    
    new_data = {}

    for d in data:
        if data[d]['coordinate group'] == 201:
            new_data[d] = data[d]
    

    with open( f_name_r, 'w' ) as f:
        f.write( json.dumps( new_data, indent=4 ) )


if __name__=='__main__':
    trim_reduced()