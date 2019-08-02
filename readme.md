# Predicting Zoning Districts from Digitized Urban Features (Austin, TX)
The City of Austin GIS department has been manually digitizing and labeling urban features for the past several years. This data is immensely valuble and costly and is essentially a gold mine for urban analytics and city planning. For my project, I wanted to use these urban features to try and predict city zoning districts (Residential, Commercial, Industrial, and Special Purpose) using various data mining and machine learning algorithms.

## Data, Methods, and Results
![alt text](https://github.com/ledanieln/CoA-Zone-Prediction/blob/master/AustinUrbanFeatures.png "Urban Feature Data from Austin GIS Department")  
Urban Feature Data from City of Austin GIS Department - Figure Generated in QGIS

The processed dataset contains |Y| = 4 labels  (Residential, Commercial, Industrial, and Special Purpose) and |X| = 60 classes or urban features. The data was input through numerous models that have varying accuracy for each class. The models predict Residential, Commercial, and Special Purpose zones well (>= 60% accuracy) depending on the model. The ensemble method combining Random Forest, Decision Tree (information gain), and K Nearest Neighbors was the most balanced classifier. 

# Tools 
**(QGIS, Python: pandas, numPy, Jupyter, matplotlib, scikit-learn)**  
I used the open source GIS software, QGIS, for producing spatial queries and areal statistics. I tried inputting into a PostgreSQL database, but it took more work to get the correct schema and getting rid of Multipolygons, 3D Multipolygons, and other artifacts of the manual digitization. QGIS offers a good way to perform GUI based data processing techniques on both vector and raster data. 
I used Python for all other data manipulation and processing techniques. Pandas and numPy were used for loading data, iterating through data and conducting calculations, matplotlib and Jupyter notebook was used to visualize, explore, and graph data, and scikit-learn was used for building the models and processing the data to fit in the model.  

Most of the scripts I used are located in the *src* folder, and my Jupyter notebooks are located in the *notebooks* folder. The *models* folder contains scripts mostly using scikit-learn to generate the predictive models. Results are detailed in the powerpoint.  

This project was completed for a Data Mining class at Texas State University. For more details, my final presentation (presentation.pptx) is located in the main folder. Please contact me for a copy of the final report of the project.
