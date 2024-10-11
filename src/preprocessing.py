import pandas as pd


def cargar_datos(ruta):
    return pd.read_csv(ruta)

def forma(df):
    dimensions = df.shape
    return f"{dimensions[0]} filas y {dimensions[1]} columnas"