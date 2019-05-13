# Predicting Zoning Districts from Digitized Urban Features
This project is labeled urban features in Austin, TX to predict zoning districts. This project was completed for a Data Mining class at Texas State University.

## Background
This is one of my first attempts at developing a 'data science' project. We received labeled data from the City of Austin that digitized structures, paved roads, and other urban features. I combined the labeled data with zoning district data from the City of Austin GIS database with the hopes of being able to predict zoning districts from the labeled features.

## Motivation
For my project, I wanted to go through the entire data science process, from feature generation and feature selection, to building a model that would be able to predict zoning districts (Residential, Commercial, Industrial, Special Purpose) from digitized and labeled polygon data. With my expertise in handling spatial data and spatial analysis, I hope to develop a pipeline for future prediction of zoning from labeled polygon data.

## Data Processing
I mainly used the open source GIS software, QGIS, for producing spatial queries and areal statistics. I tried inputting into a PostgreSQL database, but it took more work to get the correct schema and getting rid of Multipolygons, 3D Multipolygons, and other artifacts of the manual digitization.

## Data Exploration

