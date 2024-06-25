# California-Encoder-Transfer
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
In order to set up the environment, you have to install the libraries of requirements.txt.

If you have already installed the anaconda, you can easily set up with
```bash
conda env create -f requirements.txt
```

## Dataset
[us_monitoring.csv](https://drive.google.com/file/d/1G_nd7PGVF51kL-PtJVYlrIQ1UBmhd0Vd/view?usp=drive_link)
[california whole daily](https://drive.google.com/file/d/1_2BYE8ARP3dN0GtQlSz7CtvDtKKNPB-w/view?usp=sharing)
[lima_monitoring.csv](https://drive.google.com/file/d/1m3vo-fdFPsI0nUxhewav0z3U-vlr4ENR/view?usp=sharing)
[Lima2016](https://drive.google.com/file/d/1hRgBhvYJ9295fPq1_pq12OCy29ra8dG_/view?usp=sharing)
