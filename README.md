# Spatial-Transfer-Learning
Code, and Supplementary for the paper: _"Spatial Transfer Learning for Estimating PM2.5 in Data-poor Regions"_ submitted to ECML PKDD 2024

## Folders Description:
1. **country_data_process:** Transforms the California-Nevada and Lima datasets for estimation on the whole area.
2. **data_compose:** Splits target monitoring stations and composes LDF-input station cloud dataset for split train-test, estimation for each country.
3. **data_process:** Data processing modules used for LDF-input station cloud composing, spliting target monitoring stations, train-test dataset split, and so on. 
4. **map plot:** Plots the estimation results on the map.
5. **model:** Contains the all model source codes.
6. **paper_plots:** Plotting codes and graph, description, and so on.
7. **result:** Result saving file is contained.
8. **sample_data:** Contains sample dataset.

## Environment Installation
In order to set up the environment, you have to install the packages of requirements.txt. <br/>
If you have installed the anaconda, you can easily set up the environment with
```bash
conda env create -f requirements.txt
```

## Dataset
Following datasets will be necessary to implement the algorithm. <br/>
[us_monitoring.csv](https://drive.google.com/file/d/1G_nd7PGVF51kL-PtJVYlrIQ1UBmhd0Vd/view?usp=drive_link) <br/>
[california whole daily](https://drive.google.com/file/d/1_2BYE8ARP3dN0GtQlSz7CtvDtKKNPB-w/view?usp=sharing) <br/>
[lima_monitoring.csv](https://drive.google.com/file/d/1m3vo-fdFPsI0nUxhewav0z3U-vlr4ENR/view?usp=sharing) <br/>
[Lima2016](https://drive.google.com/file/d/1hRgBhvYJ9295fPq1_pq12OCy29ra8dG_/view?usp=sharing) <br/>

## Data Pre-process
1. **create data_path.py:** All data path information will be referenced from data_path.py file. Therefore, you have to create data_path.py file in data_process directory (data_process/data_path.py). Following is an example of data_path.py.
```python
pred_whole_us_path = "D:/US_PM25_data/largeUS_coords_pred.csv"
country_path = {
    "lima":"D:/US_PM25_data/Lima/",
    "california":"D:/US_PM25_data/california/"
}
country_daily_data_path = {
    "usa": "D:/US_PM25_data/california/daily_whole_data/",
    "lima":"D:/US_PM25_data/Lima/Lima whole daily/",
    "california":"D:/US_PM25_data/california/california whole daily/"
}
monitoring_country_data_path = {
    "usa": "D:/US_PM25_data/us_monitoring.csv",
    "lima": "D:/US_PM25_data/Lima/lima_monitoring.csv",
    "california_whole": "D:/US_PM25_data/california/california_whole_monitoring.csv",
    "california9": "D:/US_PM25_data/california/california_station9_monitoring.csv"
}
country_compose_data_path = {
    "lima": "D:/US_PM25_data/Lima/split-data/",
    "california": "D:/US_PM25_data/california/split-data/"
}
predict_ldf_data_path = {
    "lima": "D:/US_PM25_data/Lima/daily LDF input/",
    "california":"D:/US_PM25_data/california/daily LDF input/"
}
predict_result_path = {
    "lima": "D:/US_PM25_data/Lima/predictions/",
    "california": "D:/US_PM25_data/california/predictions/"
}
```
