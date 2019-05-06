import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataPath = '../data/processed/validation.csv'
output = '../data/processed/testFeatures.csv'

train = pd.read_csv(dataPath)

structCount = []
percentStructureArea = []
dam = []
gravelSandpit = []
unpavedRoad = []
trail = []
sidewalkArea = []
structRatio = []
recreationScore = []

#Generate Rec Score
recScore = []
for i in train.index:
    oVal = train.loc[i, 'Open SpaceCount']
    uVal = train.loc[i, 'Unpaved Athletic FieldCount']
    rbVal = train.loc[i,'Recreation Court/Ball FieldCount']
    totalVal = oVal+uVal+rbVal
    recScore.append(totalVal)

#Generate Structure Ratio
structureRatio = []
for i in train.index:
    if train.loc[i, 'percentStructureArea'] == 0:
        structureRatio.append(0)
    else:
        aVal = train.loc[i, 'StructureCount']
        bVal = train.loc[i, 'percentStructureArea']
        structureRatio.append(aVal/bVal)

train['recScore'] = recScore
train['structRatio'] = structureRatio

for i in train.index:
    swVal = train.loc[i, 'Unpaved RoadCount']
    if swVal > 0:
        unpavedRoad.append(1)
    else:
        unpavedRoad.append(0)

for i in train.index:
    gsVal = train.loc[i,'Gravel/SandpitCount']
    if gsVal > 0:
        gravelSandpit.append(1)
    else:
        gravelSandpit.append(0)
        
    damVal = train.loc[i, 'DamCount']
    if damVal > 0:
        dam.append(1)
    else:
        dam.append(0)
        
    structVal = train.loc[i, 'percentStructureArea']
    if structVal > train['percentStructureArea'].mean() + train['percentStructureArea'].quantile(q=0.75):
        percentStructureArea.append(1)
    else:
        percentStructureArea.append(0)
        
    scVal = train.loc[i, 'StructureCount']
    if scVal > train['StructureCount'].mean() + train['StructureCount'].quantile(q=0.75):
        structCount.append(1)
    else:
        structCount.append(0)

for i in train.index:
    tVal = train.loc[i, 'TrailCount']
    if tVal > 0:
        trail.append(1)
    else:
        trail.append(0)

for i in train.index:
    sVal = train.loc[i, 'percentSidewalkArea']
    if sVal > (train['percentSidewalkArea'].mean() + train['percentSidewalkArea'].quantile(q=0.75)):
        sidewalkArea.append(1)
    else:
        sidewalkArea.append(0)

for i in train.index:
    sVal = train.loc[i, 'structRatio']
    if sVal > (train['structRatio'].mean() + train['structRatio'].quantile(q=0.75)):
        structRatio.append(1)
    else:
        structRatio.append(0)

for i in train.index:
    sVal = train.loc[i, 'recScore']
    if sVal > (train['recScore'].mean() + train['recScore'].quantile(q=0.75)):
        recreationScore.append(1)
    else:
        recreationScore.append(0)

newTrainData = pd.DataFrame()
newTrainData['structCount'] = structCount
newTrainData['percentStructArea'] = percentStructureArea
newTrainData['dam'] = dam
newTrainData['gravelSandpit'] = gravelSandpit
newTrainData['unpavedRoad'] = unpavedRoad
newTrainData['trail'] = trail
newTrainData['structRatio'] = structRatio
newTrainData['sidewalkArea'] = sidewalkArea
newTrainData['recScore'] = recreationScore
newTrainData['zoneType'] = train['zoneType']

newTrainData.to_csv(output, index=False)