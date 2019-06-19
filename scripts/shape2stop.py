import pandas as pd
import geopandas as gpd
import numpy as np
import sys 
from scipy.spatial.distance import cdist

# avoid some bs warnings 
pd.options.mode.chained_assignment = None


'''Arguments and File path'''

userRoute = sys.argv[1]

filepath = r'bbb-new'


''' load and save functions.'''


# load GTFS
def load(fName): 
    f = filepath + "/" + "{}.txt".format(fName)
    return pd.read_csv(f)

# save file
def save(dfName,fName='shapes_NEW'):
    df = dfName
    df.to_csv(filepath + '/' + "{}.txt".format(fName), sep =',', index=False, float_format="%.6f")
    return print("Saved {}.txt to {}".format(fName,filepath))

# load GTFS files
trips = load('trips')
stops = load('stops')
stop_times = load('stop_times')
shapes = load('shapes')
routes = load('routes')



'''Use the user input to query out trips, shapes, stops so lat lon can be used'''
singleRoute = trips.route_id == int(userRoute)


subShapeList = trips[singleRoute].shape_id.unique().tolist()
subShapes = shapes[shapes.shape_id.isin(subShapeList)]

subTripsList = trips[singleRoute].trip_id.unique().tolist()
subTrips = trips[trips.trip_id.isin(subTripsList)]

subStopTimesList = stop_times.stop_id[stop_times.trip_id.isin(subTripsList)].tolist()

subStops = stops[stops.stop_id.isin(subStopTimesList)]



'''MAIN WORK HERE'''

# turn stops coords to list, then numpy array
# this will be used for the distance function
stpLat = subStops.stop_lat.tolist()
stpLon = subStops.stop_lon.tolist()
stpCoord = np.array(list(zip(stpLat,stpLon)))

# turn shape coords to list, then turn to numpy array
# this will be used for the distance function
shpLat = subShapes.shape_pt_lat.tolist()
shpLon = subShapes.shape_pt_lon.tolist()
shpCoord = np.array(list(zip(shpLat,shpLon)))

# create a column and run the euclidean algorithum on it
subShapes['dist_to_stp'] = cdist(shpCoord,stpCoord,'euclidean').min(axis=1) * 10000



# find difference between dist_to_stp rows
subShapes['diff'] = subShapes['dist_to_stp'] - subShapes['dist_to_stp'].shift(1)
# create a dummy column marking where the diff is negative
subShapes['dummy'] = np.where(subShapes['diff'] < 0,'1','0')

'''Keep where diff == NaN- this is first shape point and 'should' begin at a stop OR where distance to stop is < 2.8 OR where dummy == 1 AND the the value below it is == 0
AND Keep where dummy == 1 AND distance to stop is < 5 (this may change). This logic will ensure the closest stop is grabbed based on criteria other than spatial criteria. 
The result is a column named "keep" with values of either keep or throw. These values are used later on with a buffer from the stop to highlight the rows to change'''

subShapes['keep'] = np.where(np.logical_or(subShapes['diff'].isna(),np.logical_or(subShapes['dist_to_stp'] < 2.8,np.logical_and(np.logical_and(subShapes['dummy'] == '1', subShapes['dist_to_stp'] < 5.),np.logical_and(subShapes['dummy'] == '1', subShapes['dummy'].shift(-1) == '0')))) ,'keep','throw')


'''With all of the other parameters above, have a column with a buffer of the stop and check to see if a shape point marked as keep
falls within that buffer. If it does, update the shape point lat and lon for that row. There may be more than one shape point moved, inital testng 
has shown this not to be a problem. The generaliztion around the stops should not be noticeable. More testing should be done to 100% confirm'''

# create a geodataframe to do an intersection 
intersect_df = gpd.GeoDataFrame(subShapes, geometry = gpd.points_from_xy(subShapes.shape_pt_lon,subShapes.shape_pt_lat))
intersect_df['shape_pt_geometry'] = intersect_df['geometry']
intersect_df.drop(['geometry'],axis=1)

# stop dataframe with a buffer- distance can be adjusted if need be
stopdf = gpd.GeoDataFrame(subStops, geometry=gpd.points_from_xy(subStops.stop_lon,subStops.stop_lat))
stopdf['geometry'] = stopdf.geometry.buffer(.0005)

# intersect df based on buffer polygon
intersect_join = gpd.sjoin(intersect_df,stopdf,how='inner',op='intersects')
# only keep the records 'keep'
intersect_join = intersect_join[(intersect_join.keep == 'keep')]

'''join the intersect df back to the nearest shape df based on index and create the new shapes.txt'''
badMerge = pd.merge(subShapes,intersect_join,how='left',suffixes=('','_y'),left_index=True,right_index=True)
# got a duplicated index somehow, drop it here
finalMerge = badMerge[~badMerge.index.duplicated()]

# update the new join shape lat lon with stop lat lon based on the 'keep' column
finalMerge.loc[finalMerge['keep'] == 'keep', 'shape_pt_lat'] = finalMerge['stop_lat']
finalMerge.loc[finalMerge['keep'] == 'keep', 'shape_pt_lon'] = finalMerge['stop_lon']

# merge the lat lon 
finalMerge.round({'shape_pt_lat': 6, 'shape_pt_lon': 6})

# drop unwanted columns from the final merge df to create the shapes_NEW.txt
dropCols = [i for i in range(len(finalMerge.columns)) if i > 4]
finalMerge.drop(finalMerge.columns[dropCols],axis=1,inplace=True)


#  print the amount of points that were changed
changedPoints = finalMerge.shape_pt_lat != subShapes.shape_pt_lat
print('SWEET! You modified {} shape points.\n'.format(len(finalMerge[changedPoints])))

# save new shape_NEW.txt file
save(finalMerge)