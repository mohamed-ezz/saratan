import os

cwd=os.getcwd()


params = dict()
params['DataManagerParams']=dict()
params['ModelParams']=dict()

#params of the algorithm
params['ModelParams']['numcontrolpoints']=2
params['ModelParams']['sigma']=15
params['ModelParams']['device']=0
params['ModelParams']['prototxtTrain']=os.path.join(cwd,'Prototxt/train_noPooling_ResNet_cinque.prototxt')
params['ModelParams']['prototxtTest']=os.path.join(cwd,'Prototxt/test_noPooling_ResNet_cinque.prototxt')
params['ModelParams']['snapshot']=0
params['ModelParams']['dirTrain']='/data/3dircad'
#params['ModelParams']['dirTrain']='/data/3dircad/tmp'
params['ModelParams']['dirTest']='/data/3dircad/test'
params['ModelParams']['dirResult']=os.path.join(basePath,'results') #where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots']=os.path.join(basePath,'models/3dircad/') #where to save the models while training
params['ModelParams']['batchsize'] = 1 #the batchsize
params['ModelParams']['numIterations'] = 10000 #the number of iterations
params['ModelParams']['baseLR'] = 0.0001 #the learning rate, initial one
params['ModelParams']['nProc'] = 3 #the number of threads to do data augmentation


#params of the DataManager
params['DataManagerParams']['dstRes'] = np.asarray([4,4,3],dtype=float)
params['DataManagerParams']['VolSize'] = np.asarray([128,128,64],dtype=int)
params['DataManagerParams']['normDir'] = False #if rotates the volume according to its transformation in the mhd file. Not reccommended.