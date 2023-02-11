import numpy as np
import main_fed_hetero

for iid in [1,3,2]:    
    for flag in [1,2,3,4,5]:
        main_fed_hetero.mainfunc(flag,iid)