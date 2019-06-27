import pandas as pd
import geopandas as gpd
import numpy as np
import sys 
from scipy.spatial.distance import cdist

# avoid some warnings 
pd.options.mode.chained_assignment = None

filepath = sys.argv[1]

def load(fName): 
    f = filepath + "/" + "{}.txt".format(fName)
    return pd.read_csv(f)

def save(dfName,fName='shapes_NEW'):
    df = dfName
    df.to_csv(filepath + '/' + "{}.txt".format(fName), sep =',', index=False, float_format="%.6f")
    return print("Saved {}.txt to {}".format(fName,filepath))

# load GTFS files
try:
    trips = load('trips')
    stops = load('stops')
    stop_times = load('stop_times')
    shapes = load('shapes')
    print('\nAll needed GTFS files found.')
except:
    print('\nERROR....\nWhere is your GTFS?\n')
    sys.exit(1)


# may break this out into small functions later
def shape2stop(trips,shapes,stop_times,stops):

    shapeList = trips.shape_id.unique().tolist()
    data = []
    totalChanged = []
    for i in shapeList:
        subShapes = shapes[shapes.shape_id == i]
        subTripsList = trips[trips.shape_id == i].trip_id.unique().tolist()
        subTrips = trips[trips.trip_id.isin(subTripsList)]
        subStopTimesList = stop_times.stop_id[stop_times.trip_id.isin(subTripsList)].tolist()
        subStops = stops[stops.stop_id.isin(subStopTimesList)]
        
        # turn stops & shapes coords to list, then numpy array
        # this will be used for the distance function
        stpLat = subStops.stop_lat.tolist()
        stpLon = subStops.stop_lon.tolist()
        stpCoord = np.array(list(zip(stpLat,stpLon)))

        shpLat = subShapes.shape_pt_lat.tolist()
        shpLon = subShapes.shape_pt_lon.tolist()
        shpCoord = np.array(list(zip(shpLat,shpLon)))
        
        # get distance of each shape point from nearest stop
        subShapes['dist_to_stp'] = cdist(shpCoord,stpCoord,'euclidean').min(axis=1) * 10000


        # find difference between dist_to_stp rows
        subShapes['diff'] = subShapes['dist_to_stp'] - subShapes['dist_to_stp'].shift(1)

        # create a dummy column marking where the diff is negative
        subShapes['dummy'] = np.where(subShapes['diff'] < 0,1,0)

        # logic to decide which shape points to keep, o in this case, change
        subShapes['keep'] = np.where(np.logical_or(subShapes['diff'].isna(),\
                                                            np.logical_or(subShapes['dist_to_stp'] < 2.8,\
                                                                            np.logical_and(np.logical_and(subShapes['dummy'] == 1, subShapes['dist_to_stp'] < 5.0),\
                                                                                        np.logical_and(subShapes['dummy'] == 1, subShapes['dummy'].shift(-1) == 0)\
                                                                                        )\
                                                                        )\
                                                            ) ,'keep','throw')


        # create a geodataframe to do an intersection 
        intersect_df = gpd.GeoDataFrame(subShapes, geometry = gpd.points_from_xy(subShapes.shape_pt_lon,subShapes.shape_pt_lat))
        intersect_df['shape_pt_geometry'] = intersect_df['geometry']
        intersect_df.drop(['geometry'],axis=1)

        # stop dataframe with a buffer- distance can be adjusted if need be
        stopdf = gpd.GeoDataFrame(subStops, geometry=gpd.points_from_xy(subStops.stop_lon,subStops.stop_lat))
        stopdf['geometry'] = stopdf.geometry.buffer(0.000205)

        # intersect df based on buffer polygon
        intersect_join = gpd.sjoin(intersect_df,stopdf,how='inner',op='intersects')

        # create a column to identify the points that were joined
        intersect_join['joined'] = 'true'

        # join the intersect df back to the nearest shape df based on index and create the new shapes.txt
        badMerge = pd.merge(subShapes,intersect_join,how='left',suffixes=('','_y'),left_index=True,right_index=True)
        # got a duplicated index somehow, drop it here
        finalMerge = badMerge[~badMerge.index.duplicated()]
        
        # change out shape point lat and lon with stop lat and lon
        finalMerge.loc[(finalMerge['joined'] == 'true') & (finalMerge['keep'] == 'keep'), 'shape_pt_lat'] = finalMerge['stop_lat']
        finalMerge.loc[(finalMerge['joined'] == 'true' ) & (finalMerge['keep'] == 'keep'), 'shape_pt_lon'] = finalMerge['stop_lon']

        finalMerge.round({'shape_pt_lat': 6, 'shape_pt_lon': 6})

        dropCols = [i for i in range(len(finalMerge.columns)) if i > 4]
        finalMerge.drop(finalMerge.columns[dropCols],axis=1,inplace=True)

        #  print the amount of points that were changed
        changedPoints = finalMerge.shape_pt_lat != subShapes.shape_pt_lat
        percentChanged =  round((len(finalMerge[changedPoints])/len(subShapes) * 100),2)
        
        totalChanged.append(finalMerge[changedPoints])
        data.append(finalMerge)


    finalMerge = pd.concat(data)
    totalChanged = pd.concat(totalChanged)
    totalChangedPercent = round((len(totalChanged.shape_id.tolist())/len(shapes.shape_id) * 100),2)
    print('\nConnected {} ({}%) shape points to a stop'.format(len(totalChanged),totalChangedPercent))

    return save(finalMerge)


print('\nNow, moving shape points...')

# run the main function here
shape2stop(trips,shapes,stop_times,stops)



