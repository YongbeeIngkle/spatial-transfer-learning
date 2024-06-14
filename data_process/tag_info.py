## 27 + 1 features and target
usa_whole_tags = ['day', 'month', 'cmaq_x', 'cmaq_y', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'emissi11_pm25', 'pm25_value']

usa_input_tags = ['day', 'month', 'cmaq_x', 'cmaq_y', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is',
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc',
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'emissi11_pm25']

spatial_cmaq_tags = ['cmaq_x', 'cmaq_y', 'elev', 'forest_cover', 'pd', 'is', 'nldas_pevapsfc','nldas_pressfc',
    'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc','nldas_dswrfsfc',
    'nldas_pcpsfc', 'nldas_fpcsfc', 'emissi11_pm25']

transfer_tags = {
    "lima": {
        "source": ['day', 'month', 'lon', 'lat', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_pressfc', 
                'nldas_pcpsfc', 'nldas_dswrfsfc', 'elev', 'pd', 'nldas_ugrd10m', 'nldas_vgrd10m',
                'gc_aod', 'pm25_value'],
        "target": ['doy', 'month', 'Lon', 'Lat', 'temp_2m', 'rhum', 'surf_pres',
                    'conv_prec', 'short_radi_surf', 'DEM', 'Population', 'zonal_wind_10m', 'merid_wind_10m',
                    'AOD550', 'pm25_value']
    },
    "california": {
        "source": usa_whole_tags,
        "target": usa_whole_tags
    }
    }

transfer_ldf_input_tags = {
    "lima": ['day', 'month', 'lon', 'lat', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_pressfc', 'nldas_pcpsfc', 'nldas_dswrfsfc', 'elev', 
             'pd', 'nldas_ugrd10m', 'nldas_vgrd10m', 'gc_aod', 'pm25_value'],
    "california": usa_whole_tags
}

transfer_spatial_coord_tags = {
    "lima": ['lon', 'lat', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_pressfc', 'nldas_pcpsfc', 'nldas_dswrfsfc', 'elev',
             'pd', 'nldas_ugrd10m', 'nldas_vgrd10m'],
    "california": spatial_cmaq_tags
}

earth_coord_tags = {
    "lima": ["lon", "lat"],
    "california": ["cmaq_x", "cmaq_y"]
}

## target features for creating ldf_train
ldf_a_tags = ['gc_aod', 'pm25_value']
