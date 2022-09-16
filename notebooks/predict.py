import os
# zip class names into dict
labels = [  "AnnualCrop", 
            "Forest", 
            "HerbaceousVegetation", 
            "Highway", 
            "Industrial", 
            "Pasture", 
            "PermanentCrop", 
            "Residential",
            "River", 
            "SeaLake" ]

classes = dict(zip( labels,range (len(labels) ) ) )


# In[7]:


import pandas as pd

# read subset csv files into data frames
df = { 'train' : None, 'test' : None }
for subset in [ 'train', 'test' ]:
    df[ subset ] = pd.read_csv( os.path.join( os.getcwd(), 'eurosat/{}.csv'.format( subset ) ) )
    
    # update image pathnames
    # df[ subset ]['pathname'] = df[ subset ]['pathname'].apply(lambda x: 
    #                                   x.replace( os.path.dirname(x)[ 0 : os.path.dirname(x).find( repo ) + len ( repo ) ], root_path) )   
df[ 'train' ].head(5)

# 
# In[8]:


import gdal
import numpy as np
import matplotlib.pyplot as plt

# define figure content
nrows=len(classes); ncols=5

# create figure
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 24))
fig.suptitle('Test RGB Images : Sentinel-2 Bands 4, 3, 2', fontsize=18)

# plot rgb bands of randomly selected test images
test = df[ 'test']; bands = [ 4,3,2 ]
for row, c in enumerate( classes ):

    # pick random sample from each class population
    subset = test [ test[ 'class'] == c ]
    sample = subset.sample(ncols) 
    
    col = 0
    for idx, record in sample.iterrows():
        
        # use gdal to open multispectral 16-bit imagery
        ds = gdal.Open( record[ 'pathname'] )
        
        rgb = []
        for bid in bands:

            # read band data
            band = np.asarray( ds.GetRasterBand( bid ).ReadAsArray(), dtype=float )
            
            # compute 16-bit to 8bit min / max scaling factors
            r = np.percentile( band, [ 2, 98 ] )            
            band = (( band - r[0]) / (r[1] - r[0] ) ) * 255.0
            
            # clip to 8bit
            band = np.clip( band, a_min=0.0,a_max=255.0 )
            rgb.append ( np.asarray( band, dtype=np.uint8 ) )
            
        # display rgb image and filename title
        axes[ row ][ col ].imshow( np.asarray( np.dstack( rgb ) ) )
        axes[ row ][ col ].set_title( '{}'.format ( os.path.basename( record[ 'pathname' ] ) ) )
        
        # remove axes ticks
        axes[ row ][ col ].get_xaxis().set_ticks([])
        axes[ row ][ col ].get_yaxis().set_ticks([])
        
        col += 1

# tight layout with adjustment
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# raise Exception("checkpoint")  


# ## Model
# A VGG-16-based CNN - preloaded with ImageNet weights - was selected as backbone for image feature extraction. The topmost layer of VGG16 CNN was attached to two fully connected 'Relu'-layers comprising 256 and 128 units respectively with dropout for regularization. The output layer comprised a single unit with softmax activation function to implement the classification. 
# 
# ![architecture](https://raw.githubusercontent.com/chris010970/eurosat/master/notebooks/assets/architecture.png)

# ## Results
# The model was trained minimising categorical cross entropy loss function with Adam optimiser configured with a learning rate of 1e-6. High training and validation categorical accuracies (> 0.9) were reported after 100-150 epochs. Inference analysis in the form of a confusion matrix confirmed the model had acquired the capacity to accurately identify different land cover types. It should be noted that predictive accuracy for Highways class (~0.8) was suspectible to increased error where samples were incorrectly classified as 'Residential' and 'Industrial'. 

# In[9]:


# load class-specific diagnostics from csv file
path = 'vgg16-256-128/'
df[ 'log' ] = pd.read_csv( os.path.join( path, 'log.csv') )

# create figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig.suptitle('Training Diagnostics', fontsize=18)

# plot diagnostics of model 
df[ 'log' ].plot.line( ax=axes[0], y=['loss'])
df[ 'log' ].plot.line( ax=axes[0], y=['val_loss'])

# plot diagnostics of model 
df[ 'log' ].plot.line( ax=axes[1], y=['categorical_accuracy'])
df[ 'log' ].plot.line( ax=axes[1], y=['val_categorical_accuracy'])


# In[10]:


# read global band statistics
stats = pd.read_csv( os.path.join( os.getcwd(), 'eurosat/stats.csv' ) )
stats.head(13)


# In[11]:


# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

# data frame amendments
for subset in [ 'train', 'test' ]:   

    # add one hot encoder column to data frames
    df[ subset ][ 'target' ] = tuple ( to_categorical(  df[ subset ][ 'id' ], num_classes=len( classes ) ) )

df['train'].head(5)


# In[12]:


# load pre-trained model from file - downloaded from GCS
from src.model import loadFromFile
model, model_type = loadFromFile( 'vgg16-256-128' )


# In[13]:


from src.generator import MultiChannelImageDataGenerator

def getPrediction( model, df, stats ):

    """
    generate prediction for images referenced in data frame
    """

    # create generator
    batch_size = 1
    generator = MultiChannelImageDataGenerator( [ df ],
                                                batch_size,
                                                stats=stats,
                                                shuffle=False )

    # initiate prediction
    steps = len( df ) // batch_size
    y_pred = model.predict_generator( generator, steps=steps )

    # return index of maximum softmax value
    return np.argmax( y_pred, axis=1 )


# In[14]:


from sklearn.metrics import confusion_matrix

def getConfusionMatrix( actual, predict, labels ):

    """
    compute confusion matrix for prediction
    """

    # compute normalised confusion matrix 
    cm = confusion_matrix( actual, predict )
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # parse normalised confusion matrix into dataframe
    return pd.DataFrame( cm, index=labels, columns=labels )


# In[15]:


import seaborn as sn

def plotConfusionMatrix( cm, subset ):

    """
    plot train and test confusion matrix
    """

    # create figure
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    # plot heatmap - adjust font and label size
    sn.set(font_scale=1.0) 
    sn.heatmap(cm, annot=True, annot_kws={"size": 14}, fmt='.2f', ax=axes )

    axes.set_title( 'Normalised Confusion Matrix: {}'.format( subset ) )
    plt.show()

    return


# In[16]:


import numpy as np

# generate actual vs prediction
# for subset in [ 'train', 'test' ]:
for subset in [ 'test' ]:

    actual = np.asarray( df[ subset ][ 'id' ].tolist(), dtype=int )
    predict = getPrediction( model, df[ subset ], stats )

    # get confusion matrix
    cm = getConfusionMatrix( actual, predict, classes.keys() )

    # plot confusion matrix
    plotConfusionMatrix( cm, subset )


# In[ ]:




