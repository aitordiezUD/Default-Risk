import pandas as pd
from sklearn.impute import KNNImputer


def cargar_datos(ruta):
    return pd.read_csv(ruta)

def forma(df):
    dimensions = df.shape
    return f"{dimensions[0]} filas y {dimensions[1]} columnas"


def format_boolean_columns(df, boolean_columns, true_label='Y', false_label='N'):
    """
    Converts specified boolean columns from custom true/false labels to boolean True/False.

    Parameters:
        df: The DataFrame to be formatted.
        boolean_columns (str or list): A column name or a list of column names that contain custom boolean values.
        true_label: The label representing True (default is 'Y').
        false_label: The label representing False (default is 'N').

    Returns:
        pd.DataFrame: The DataFrame with corrected boolean columns.
    """
    if isinstance(boolean_columns, str):
        boolean_columns = [boolean_columns]

    for col in boolean_columns:
        df[col] = df[col].map({true_label: True, false_label: False})

    return df


def missing_values_percentage(df):
    """
    Calculates the percentage of missing values in each column of a DataFrame.
    Args:
        df: the df frame to analyze

    Returns:
        pd.DataFrame: A DataFrame containing the percentage of missing values in each column.
    """

    missing_percent = df.isnull().mean() * 100
    missing_df = pd.DataFrame(missing_percent, columns=['Missing Percentage'])
    missing_df = missing_df[missing_df['Missing Percentage'] > 0]
    missing_df = missing_df.sort_values(by='Missing Percentage', ascending=False)

    return missing_df

def drop_columns(df, missing_df, default_threshold=50):
    """
    Drops columns with missing values above a certain threshold.

    Args:
        df: DataFrame to drop columns from.
        missing_df: DataFrame with the percentage of missing values in each column
        default_threshold: The threshold above which columns will be dropped (default is 50).

    Returns:
        pd.DataFrame: df DataFrame with columns dropped.
    """
    columns_to_drop = missing_df[missing_df['Missing Percentage'] > default_threshold].index
    df = df.drop(columns=columns_to_drop)

    return df



def impute_with_knn(df, n_neighbors=5):
    """
    Imputes missing values using the K-Nearest Neighbors Imputer algorithm.

    Args:
        df: DataFrame to impute missing values to.
        columns_to_impute: Columns to impute missing values to.
        n_neighbors: Number of neighbors to use for imputation (default is 5).

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    return df_imputed


