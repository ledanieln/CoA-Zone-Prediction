import pandas as pd
from progress.bar import IncrementalBar

#Import Data
dfnorth = pd.read_csv('../data/processed/nAustinContains.csv')
dfsouth = pd.read_csv('../data/processed/sAustinContains.csv')

#Generate list of unique zone types
zoneList = list(dfnorth.ZONING_ZTY.unique())
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
dfnorth['Reduced-Zone'] = dfnorth['ZONING_ZTY'].map(zoneMap)

#Create dictionary for clean dataset
train = {}

uniqueZoneList = list(dfnorth.ZONING_ID.unique())

#Make sure the training and testing datasets contain same number of features
uniqueNFeatureList = list(dfnorth.FEATURE.unique())
uniqueSFeatureList = list(dfsouth.FEATURE.unique())
uniqueFeatureList = set(uniqueSFeatureList) & set (uniqueNFeatureList)

#Create entry for each zone
bar = IncrementalBar('Creating base dictionary..', max = len(uniqueZoneList))
for zone in uniqueZoneList:
    zoneDict = {}
    zoneDict['zoneID'] = int(zone)
    zoneTable = dfnorth[dfnorth['ZONING_ID']==zone].reset_index(drop=True)
    zoneDict['zoneArea'] = zoneTable['Shape_Area'][0]
    for feature in uniqueFeatureList:
        zoneDict[feature + 'Count'] = 0
        zoneDict['percent' + feature + 'Area'] = 0
    zoneDict['zoneType'] = zoneTable['Reduced-Zone'][0]
    train[zone] = zoneDict
    bar.next()
bar.finish()

#Remove features that the area is greater than the Zone Area
# bar = IncrementalBar('Removing Large Areas..', max = len(dfnorth.index))
# for i in dfnorth.index:
#     if dfnorth.loc[i, 'z.Shape_Ar'] > dfnorth.loc[i, 'Shape_Area']:
#         dfnorth.drop([i])
#     bar.next()
# bar.finish()

#Add areal statistics for each zone
bar = IncrementalBar('Adding areal statistics..', max = len(uniqueFeatureList))
for feature in uniqueFeatureList:
    featureSums = dict(dfnorth[dfnorth['FEATURE']==feature].groupby('ZONING_ID')['Shape_Ar_1'].sum())
    featureCounts = dict(dfnorth[dfnorth['FEATURE']==feature].groupby('ZONING_ID')['Shape_Ar_1'].count())
    for key, value in featureSums.items():
        train[key]['percent' + feature + 'Area'] = value/(train[key]['zoneArea'])
    for key, value in featureCounts.items():
        train[key][feature + 'Count'] = value
    bar.next()
bar.finish()

trainSet = pd.DataFrame.from_dict(train, orient='index').reset_index(drop=True)
trainSet.to_csv('../data/processed/train2017-contains.csv', index=False)
#Features
#Count of Feature
#Percent Area of Feature
#Zone area
