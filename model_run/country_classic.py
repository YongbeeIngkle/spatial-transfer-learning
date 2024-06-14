import os
import pandas as pd
from model.classic import CountryTrainPred
from result.save import save_pred

def _train_predict(country_name):
    model_train_test = CountryTrainPred(model_name, input_normalize)
    monitoring_data = pd.read_csv(f"D:/US_PM25_data/{country_name}/{country_name}_whole_monitoring.csv")
    model_train_test.train(monitoring_data)

    data_dir = f"D:/US_PM25_data/{country_name}/{country_name} whole daily/"
    pred_files = os.listdir(data_dir)
    for dt_file in pred_files:
        pred_data = pd.read_csv(f"{data_dir}{dt_file}")
        pred_cmaq = pred_data[["cmaq_x", "cmaq_y"]]
        pred_result = model_train_test.predict(pred_data)
        save_pred(pred_cmaq, pred_result, prediction_folder, dt_file)

if __name__ == "__main__":
    country_name = "california"
    model_name = "GBR" ## RF, GBR, Kriging
    input_normalize = True

    prediction_folder = f"D:/US_PM25_data/{country_name}/predictions/{model_name}/"
    _train_predict(country_name)
