import numpy as np

def preprocess(x_numpy,windowSize):    
    #Now we are going to pre-process the x
    x_raw = np.array([x_numpy])
    '''                                                                                                                                                                                                   
    Going through the instance of data, creating arrays of windowSize          
    given and applying flatten over each list - takes around 40 seconds                                                                                                                                    
    for train.                                                 
    '''
    listOfLists = map(lambda inst: zip(*(inst[i:] for i
                                         in range(windowSize))),x_raw)
    x = np.array([item for sublist in listOfLists for item in sublist])
    #reshaping to flatten images to 40 * windowSize sized vector
    num_pixels = x.shape[1] * x.shape[2]
    x = x.reshape(x.shape[0], num_pixels).astype('float32')
    return x
        
