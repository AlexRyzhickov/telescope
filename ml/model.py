import yaml
import joblib

from pathlib import Path

# load config file
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def load_models():
    model_1 = joblib.load(config["model_1_path"])
    model_2 = joblib.load(config["model_2_path"])
    model_3 = joblib.load(config["model_3_path"])

    def predict(
            wind_speed: float,
            temp: float,
            humidity: float,
            temp1: float,
            temp2: float,
            temp3: float,
            temp4: float,
            temp5: float,
            temp6: float,
            temp7: float,
            temp8: float,
            temp9: float,
            temp10: float,
            temp11: float,
            temp12: float,
            temp13: float,
            temp14: float,
            strain1: float,
            strain2: float,
            strain3: float,
            strain4: float,
            strain5: float,
            strain6: float,
            strain7: float,
            strain8: float,
            strain9: float,
            strain10: float,
            strain11: float,
            strain12: float,
            strain13: float,
            strain14: float,
    ) -> (float, float, float):

        X = [[
            wind_speed,
            temp,
            humidity,
            temp1,
            temp2,
            temp3,
            temp4,
            temp5,
            temp6,
            temp7,
            temp8,
            temp9,
            temp10,
            temp11,
            temp12,
            temp13,
            temp14,
            strain1,
            strain2,
            strain3,
            strain4,
            strain5,
            strain6,
            strain7,
            strain8,
            strain9,
            strain10,
            strain11,
            strain12,
            strain13,
            strain14,
        ]]
        delta_a = model_1.predict(X)
        delta_e = model_2.predict(X)
        roll = model_3.predict(X)

        return (delta_a, delta_e, roll)

    return predict
