import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataPath = '../data/processed/train.csv'

def categorizeData(dataPath):
	data = pd.read_csv(dataPath)
	dataLabels = list(data)
	
	data[dataLabels[0]]
	plt.hist(data[dataLabels[0]], color='blue', edgecolor='black')

categorizeData(dataPath)