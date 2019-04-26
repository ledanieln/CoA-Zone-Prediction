import pandas as pd
from progress.bar import IncrementalBar

#Import Data
dfsouth = pd.read_csv('../data/processed/Austin-South-Features.csv')

#print(dfsouth.head())
#print(dfsouth.head())
zoneList = list(dfsouth.ZONING_ZTY.unique())
zoneMap = {}

#Map zone code to reduced zones
for row in zoneList:
    zoneHeader = row.split('-')[0]
    if zoneHeader == "LA" or zoneHeader == "RR" or zoneHeader == "SF" or zoneHeader == "MF" or zoneHeader == "MH":
        zoneMap[row] = "Residential"
    elif zoneHeader == "NO" or zoneHeader == "LO" or zoneHeader == "GO" or zoneHeader == "CR" or zoneHeader == "LR" or zoneHeader == "GR" or zoneHeader == "L"  or zoneHeader == "CBD" or zoneHeader == "DMU" or zoneHeader == "W/LO" or zoneHeader == "CS" or zoneHeader == "CS-1" or zoneHeader == "CH":
        zoneMap[row] = "Commercial"
    elif zoneHeader == "IP" or zoneHeader == "LI" or zoneHeader == "MI" or zoneHeader == "R&D":
        zoneMap[row] = "Industrial"
    else:
        zoneMap[row] = "Special Purpose"

#Add reduced zones to dataFrame
dfsouth['Reduced-Zone'] = dfsouth['ZONING_ZTY'].map(zoneMap)

print(len(dfsouth))

#Create dictionary for clean dataset
train = {}

uniqueZoneList = list(dfsouth.ZONING_ID.unique())
uniqueFeatureList = list(dfsouth.fJoin_FEATURE.unique())

#Create entry for each zone
bar = IncrementalBar('Creating base dictionary..', max = len(uniqueZoneList))
for zone in uniqueZoneList:
    zoneDict = {}
    zoneDict['zoneID'] = int(zone)
    zoneTable = dfsouth[dfsouth['ZONING_ID']==zone].reset_index(drop=True)
    zoneDict['zoneArea'] = zoneTable['Shape_Area'][0]
    for feature in uniqueFeatureList:
        zoneDict[feature + 'Count'] = 0
        zoneDict['percent' + feature + 'Area'] = 0
    zoneDict['zoneType'] = zoneTable['Reduced-Zone'][0]
    train[zone] = zoneDict
    bar.next()
bar.finish()

#Add areal statistics for each zone
bar = IncrementalBar('Adding areal statistics..', max = len(uniqueFeatureList))
for feature in uniqueFeatureList:
    featureSums = dict(dfsouth[dfsouth['fJoin_FEATURE']==feature].groupby('ZONING_ID')['fJoin_Shape_Area'].sum())
    featureCounts = dict(dfsouth[dfsouth['fJoin_FEATURE']==feature].groupby('ZONING_ID')['fJoin_Shape_Area'].count())
    for key, value in featureSums.items():
        train[key]['percent' + feature + 'Area'] = value/(train[key]['zoneArea'])
    for key, value in featureCounts.items():
        train[key][feature + 'Count'] = value
    bar.next()
bar.finish()

trainSet = pd.DataFrame.from_dict(train, orient='index').reset_index(drop=True)
trainSet.to_csv('../data/processed/validation.csv', index=False)
#Features
#Count of Feature
#Percent Area of Feature
#Zone area
