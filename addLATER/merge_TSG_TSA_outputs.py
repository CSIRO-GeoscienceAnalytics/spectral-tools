import plotly.express as px
from Ancillary.ancillary import get_average_downhole_values, get_average_values_df, ternary_color
from Ancillary.spherical_projection import SphericalProjection
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from shapely.geometry import Point, Polygon
import Ancillary.ancillary as anc
import pandas as pd
from pathlib import Path


def enact_closure(df_in):
    """

    :param df_in: incoming data frame to enforce closure with e.g. sum to 1 across a given row
    :type df_in: pandas dataframe
    :return: the dataframe with closure enacted
    :rtype: pandas dataframe
    """
    return df_in.div(df_in.sum(axis=1), axis=0)


def merge_outputs(df1_in, df2_in, drop=None, average_span=1):
    """

    :param df1_in: the first dataframe
    :type df1_in: pandas dataframe
    :param df2_in: the second dataframe
    :type df2_in: pandas dataframe
    :param drop: A list of columns names to drop from the merged files
    :type drop: list of strings
    :param average_span: the spatial averaging span in metres
    :type average_span: int
    :return: a merged spatially averaged dataframe, a meraged non-spatially averaged dataframe
    :rtype: pandas dataframe, pandas dataframe
    """
    # set the index as the depths
    df1 = df1_in.set_index(["Depth (m)"])
    df2 = df2_in.set_index(["Depth (m)"])

    # concatenate the 2 dataframes
    df = pd.concat([df1, df2], axis=1).fillna(0)

    # group by the columns and use the minimum if a column has repeat entries
    # the minimum is used as it generally means that other mineral groups for a given sample are present as well and to
    # me that means that the lower value (if 2 such values exist in 2 columns) is actually "truer"
    df = df.groupby(by=df.columns, axis=1).min()

    # add the depth back in
    df["Depth (m)"] = df.index

    # pop the depth column and relocate it to the front of the columns
    col = df.pop("Depth (m)")
    df.insert(0, "Depth (m)", col)

    # reset the index
    df = df.reset_index(drop=True)

    # if a list of columns/groups to drop has been flagged then do so
    if drop:
        for val in drop:
            if val in df.columns:
                df = df.drop(drop, axis=1)

    # get the average downhole values
    df_avg, _, _ = anc.get_average_values_df(df, step_size=average_span, field="Depth (m)")
    df_avg.iloc[:, 1:] = df_avg.iloc[:, 1:].div(df_avg.iloc[:, 1:].sum(axis=1), axis=0)

    # enforce closure of what is being returned
    df[df.columns.tolist()[1:]] = enact_closure(df.iloc[:, 1:])
    df_avg[df_avg.columns.tolist()[1:]] = enact_closure(df_avg.iloc[:, 1:])

    # return the result
    return df_avg, df

