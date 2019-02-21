import numpy as np

def ep(act_dim, obs_dim):    
    act_low = np.array([15.,5.8,10.,4.], dtype=np.float32)
    act_high = np.array([18.5,7.,18.,6.5], dtype=np.float32)
    # act_low = np.array([10,10,10,10,3,10,10,10,10,3], dtype=np.float32)
    # act_high = np.array([19,19,19,19,8,19,19,19,19,8], dtype=np.float32)
    
    obs_low = np.array([1,1,1,-22,23.4,55502,23.4,49755,5028], dtype=np.float32)
    obs_high = np.array([12,31,24,35,24.3,55540,24.3,49790,56000], dtype=np.float32)

    assert (act_dim == act_low.shape) and (act_dim == act_high.shape), "action shape mismatching"
    assert (obs_dim == obs_low.shape) and (obs_dim == obs_high.shape), "observation shape mismatching"

    return act_low, act_high, obs_low, obs_high