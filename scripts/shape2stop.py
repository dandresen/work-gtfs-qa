import pandas as pd
import geopandas as gpd
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.distance import cdist

"""Shape2Stop: (V0.1)
Author: dandresen
Date: 6/28/19

Replaces the latitude and longitude for shape points "near" a stop with the stop's latitude and longitude.

Problems fixed:
1)  Most shape issues seen in NB1 related to shape points being too far from a stop.

Process:
1)  Pulls out unique shapes with their stops and runs an euclidean distance function.
2)  Adds the following columns the a new shapes DataFrame:
        [dist_to_stp] uses the euclidean distance from a shape point to the nearest stop
        [diff] finds the difference between each row in dist_to_stp
        [dummy] finds where diff < 0 and gives it a value of 1 or 0
        [keep] uses multiple Boolean expressions based on different criteria from dummy  and dist_to_stp creating values of 'keep' or 'throw'
3)  Uses a spatial buffer around the stops and intersect the shape points.
4)  If criteria is met, replace the shape latitude and longitude with the stop latitude and longitude.
5)  The output will show the amount and percentage of shape points changed.

Instructions:
1)  Pass a folder path to the GTFS via the -p command line argument (e.g. "python shape2stop.py -p <folder-path>").
2)  Optionally -v (--verbose) for additional information on shape points for each shape_id or -r (--replace) to overwrite original shapes.txt.
3)  If not -r shapes_NEW.txt will be created.
"""

# avoid some warnings from pandas
pd.options.mode.chained_assignment = None

# check directory path
def dir_path(path):
    if Path(path).is_dir():
        return Path(path)
    else:
        raise NotADirectoryError(path)
        
# parsing arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=dir_path, help='file path to GTFS <required>', required=True)
parser.add_argument('-v', '--verbose', action="store_true", help='console log the amount of shape points that changed for each shape_id')
parser.add_argument('-o', '--overwrite', action="store_true", help='overwrite existing shapes.txt')
args = parser.parse_args()


def load(fName): 
    f = dir_path(args.path) / "{}.txt".format(fName)
    return pd.read_csv(f)

def save(dfName,fName='shapes_NEW'):
    df = dfName
    df.to_csv(dir_path(args.path) / "{}.txt".format(fName), sep =',', index=False, float_format="%.6f")
    return print("Saved {}.txt to {}".format(fName,args.path))

# load GTFS files as DataFrames
try:
    trips = load('trips')
    stops = load('stops')
    stop_times = load('stop_times')
    shapes = load('shapes')
    print('\nAll required GTFS files found\nNow, moving shape points...')
except:
    print('\nError loading your GTFS\n')

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

        # only keep needed columns
        dropCols = [i for i in range(len(finalMerge.columns)) if i > 4]
        finalMerge.drop(finalMerge.columns[dropCols],axis=1,inplace=True)

        # print the amount of points that were changed
        changedPoints = finalMerge.shape_pt_lat != subShapes.shape_pt_lat
        percentChanged =  round((len(finalMerge[changedPoints])/len(subShapes) * 100),2)
        
        # verbose output (optional)
        if args.verbose:
            print('Moved {} shape points for shape_id {} ({}%)'.format(len(finalMerge[changedPoints]),i,percentChanged)) 

        totalChanged.append(finalMerge[changedPoints])
        data.append(finalMerge)


    finalMerge = pd.concat(data)
    
    # get some information about how many shape points were moved
    totalChanged = pd.concat(totalChanged)
    totalChangedPercent = round((len(totalChanged.shape_id.tolist())/len(shapes.shape_id) * 100),2)
    print('\nConnected {} ({}%) shape points to a stop'.format(len(totalChanged),totalChangedPercent))
    
    # overwrite existing shapes.txt (optional)
    if args.overwrite:
        return save(finalMerge,'shapes')

    return save(finalMerge)

# run the main function 
shape2stop(trips,shapes,stop_times,stops)