from pyexpat import native_encoding
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.dates import DateFormatter
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.histograms import _ravel_and_check_weights
from scipy import stats

#import kriging results for later comparisons
rfd_krig = pd.read_csv("C:\\Users\\Sakhayaan Gavrilyev\\Documents\\GIS data\\radon_msk\\lognormal ord5kv2.xyz", sep = " ", header = None, names = ['X', 'Y', 'rfd'])
#import actual RFD results for training purposes
rfd_real_diff = pd.read_csv("C:\\Users\\Sakhayaan Gavrilyev\\Documents\\GIS data\\radon_msk\\rfd_diffusive_full.csv", sep = ",")
rfd_real_full = pd.read_csv("C:\\Users\\Sakhayaan Gavrilyev\\Documents\\GIS data\\radon_msk\\rfd_all.csv", sep = ",")
rasstrel = ['fid', 'vertex_index', 'vertex_part', 'vertex_part_index', 'distance', 'angle']
rfd_real_diff = rfd_real_diff.drop(rasstrel, axis = 1)
rfd_real_full = rfd_real_full.drop(rasstrel, axis = 1)
#import all the infolayers
dose = pd.read_csv("msk_doserate_2k_clip.xyz", sep = " ", header = None, names = ['X', 'Y', 'dose'], na_values = '-9999')
ra226 = pd.read_csv("msk_Ra226_real_clip.xyz", sep = " ", header = None, names = ['X', 'Y', 'ra'], na_values = '-9999')
carb = pd.read_csv("msk_carbon_2k_clip.xyz", sep = " ", header = None, names = ['X', 'Y', 'carb'], na_values = '-9999')
mezo = pd.read_csv("msk_mezo_2k_clip.xyz", sep = " ", header = None, names = ['X', 'Y', 'mezo'], na_values = '-9999')
relf = pd.read_csv("msk_quart_2k_clip.xyz", sep = " ", header = None, names = ['X', 'Y', 'hght'], na_values = '-9999')
carblayers = pd.read_csv("msk_carbonlayers_2k.xyz", sep = " ", header = None, names = ['X', 'Y', 'carb'], na_values = '-9999')
mezolayers = pd.read_csv("msk_mezolayers_2k.xyz", sep = " ", header = None, names = ['X', 'Y', 'mezo'], na_values = '-9999')
quartlayers = pd.read_csv("msk_quartlayers2k.xyz", sep = " ", header = None, names = ['X', 'Y', 'hght'], na_values = '-9999')
gdfdose = gpd.GeoDataFrame(dose, geometry = gpd.points_from_xy(dose['X'], dose['Y']))
gdfcarb = gpd.GeoDataFrame(carb, geometry = gpd.points_from_xy(carb['X'], carb['Y']))
gdfmezo = gpd.GeoDataFrame(mezo, geometry = gpd.points_from_xy(mezo['X'], mezo['Y']))
gdfrelf = gpd.GeoDataFrame(relf, geometry = gpd.points_from_xy(relf['X'], relf['Y']))
gdfcarbl = gpd.GeoDataFrame(carblayers, geometry = gpd.points_from_xy(carblayers['X'], carblayers['Y']))
gdfmezol = gpd.GeoDataFrame(mezolayers, geometry = gpd.points_from_xy(mezolayers['X'], mezolayers['Y']))
gdfquartl = gpd.GeoDataFrame(quartlayers, geometry = gpd.points_from_xy(quartlayers['X'], quartlayers['Y']))
gdfra226 = gpd.GeoDataFrame(ra226, geometry = gpd.points_from_xy(ra226['X'], ra226['Y']))
gdfrfdk = gpd.GeoDataFrame(rfd_krig, geometry = gpd.points_from_xy(rfd_krig['X'], rfd_krig['Y']))
gdfrfdd = gpd.GeoDataFrame(rfd_real_diff, geometry = gpd.points_from_xy(rfd_real_diff['X'], rfd_real_diff['Y']))
gdfrfdf = gpd.GeoDataFrame(rfd_real_full, geometry = gpd.points_from_xy(rfd_real_full['X'], rfd_real_full['Y']))


#resolution
res = 40
#merging all the infolayers into a single dataframe
gdf = gdfdose.sjoin_nearest(gdfcarb, how = "left", max_distance=res)
rasstrel2 = ['X_right', 'Y_right', 'geometry']
gdf = gdf.drop(['index_right'], axis = 1)
gdf = gdf.sjoin_nearest(gdfmezo, how = 'left', max_distance=res)
gdf = gdf.drop(['index_right'], axis = 1)
gdf = gdf.sjoin_nearest(gdfrelf, how = 'left', max_distance=res)
gdf = gdf.drop(['index_right'], axis = 1)
gdf = gdf.sjoin_nearest(gdfra226, how = 'left', max_distance=res)
gdf = gdf.drop(['index_right'], axis = 1)
gdf = gdf.sjoin_nearest(gdfcarbl, how = 'left', max_distance=res)
gdf = gdf.drop(['index_right'], axis = 1)
gdf = gdf.sjoin_nearest(gdfmezol, how = 'left', max_distance=res)
gdf = gdf.drop(['index_right'], axis = 1)
gdf = gdf.sjoin_nearest(gdfquartl, how = 'left', max_distance=res)
gdf = gdf.drop(['index_right'], axis = 1)
#diffusive rfd obtained using kriging
gdf = gdf.sjoin_nearest(gdfrfdk, how = 'left', max_distance=150)
gdf = gdf.drop(['index_right'], axis = 1)
gdf = gdf.dropna()
#normal discrete rfd measurements (diffusive only)
gdf = gdf.sjoin_nearest(gdfrfdd, how = 'left', max_distance=150)
gdf = gdf.drop(['index_right'], axis = 1)
#all discrete rfd measurements (with convective outliers)
gdf = gdf.sjoin_nearest(gdfrfdf, how = 'left', max_distance=150)
gdf = gdf.drop(['index_right'], axis = 1)
gdf = gdf.drop(rasstrel2, axis = 1)
gdf.columns = ['X', 'Y', 'dose', 'carb', 'xl', 'yl', 'mezo', 'hght', 'xl', 'yl', 'ra', 'carb_l', 'xl', 'yl', 'mezo_l', 'quart_l', 'xl', 'yl', 'rfdk', 'rfdd', 'xl', 'yl', 'rfd']
gdf = gdf.drop(['xl', 'yl'], axis = 1)
#Preparing data
df = gdf.dropna()
df = df.drop('rfdk', axis = 1)
df = df.drop('rfd', axis = 1)
df_trn = df.sample(frac=0.8)
df_tst = df.drop(df_trn.index)
ftr_trn = df_trn.copy()
ftr_tst = df_tst.copy()
lbl_trn = ftr_trn.pop('rfdd')
lbl_tst = ftr_tst.pop('rfdd')
x = df_trn.pop('X')
y = df_trn.pop('Y')
#Normalizing data and assembling the model
normalizer = preprocessing.Normalization(axis = -1)
normalizer.adapt(np.array(ftr_trn))

Nepochs = 10000

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 0.2])
    plt.xlabel('Epoch')
    plt.ylabel('Error [t]')
    plt.legend()
    plt.grid(True)

#define the model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(512, activation='sigmoid'),
        layers.Dense(512, activation='sigmoid'),
        layers.Dense(256, activation='sigmoid'),
        layers.Dense(1)
    ])
    model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model(normalizer)
#dnn_model.summary()
history = dnn_model.fit(
    ftr_trn, lbl_trn,
    batch_size = 1024,
    validation_split=0.2,
    verbose=0, epochs=Nepochs,
)
#Results
plot_loss(history)
plt.show()
predx = dnn_model.predict(ftr_tst).flatten()
a = plt.axes(aspect='equal')
plt.scatter(lbl_tst, predx)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.rcParams['figure.figsize'] = [6, 6]
plt.show()

error = predx - lbl_tst
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
_ = plt.ylabel('Count')
plt.rcParams['figure.figsize'] = [6, 6]
plt.show()

df_pred = gdf.drop('rfdk', axis = 1)
df_pred = df_pred.drop('rfd', axis = 1)
df_pred = df_pred[df_pred.rfdd.isin(df.rfdd) == False]
df_pred = df_pred.drop('rfdd', axis = 1)
predix = dnn_model.predict(df_pred).flatten()
df_pred['rfd_pred'] = predix
df_pred = df_pred[(np.abs(stats.zscore(df_pred['rfd_pred'])) < 3)]
df_pred.to_csv('prediction10k.csv', sep = ' ', header = False, index = False)
