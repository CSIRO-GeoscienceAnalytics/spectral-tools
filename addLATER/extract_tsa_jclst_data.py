from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from spectraltools.io import \
    parse_tsg  # or get it from https://github.com/FractalGeoAnalytics/pytsg


def get_average_values_df(df_in: pd.DataFrame, step_size: int=1, field: str=None):
    """
    Get the average downhoile values in the tsg dataset

    Args:
        df_in (pd.DataFrame): incoming dataframe containing data from a tsg dataset
        step_size (int, optional):Averaging range (assumed its in metres). Defaults to 1.
        field (str, optional): What field to do the averging over (usually you might want "Depth (m)"). Defaults to None.
    """
    def from_to(df_series: pd.Series, step: int) -> tuple[pd.Series, pd.Series]:
        """
        get from and to depths for each of the averged sections

        """
        start = df_series[np.unique(df_series // step, return_index=True)[1]]
        stop = pd.concat([df_series[np.unique(df_series // step, return_index=True)[1][1:] - 1],
                          pd.Series(df_series.values[-1])])
        return start, stop

    if field is None:
        section_start, section_stop = from_to(df_in.index, step_size)
        result_out = df_in.groupby(df_in.index // step_size).mean()
    else:
        section_start, section_stop = from_to(df_in[field], step_size)
        result_out = df_in.groupby(df_in[field] // step_size).mean()
    
    return result_out, section_start, section_stop

def get_average_downhole_values(depths: NDArray, y: NDArray, step_size: int=1) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    A wrapper to get the averged downhole tsg data

    Args:
        depths (NDArray): The depths corresponding to incoming samples y
        y (NDArray): An array of values that you want averaged
        step_size (int, optional): What averging space you want. Defaults to 1.

    Returns:
        tuple (pd.DataFrame, pd.Series, pd.Series): averaged y values, start location, end location
    """
    tdf = pd.concat([pd.DataFrame(depths, columns=["Depth"]), pd.DataFrame(y)], axis=1)
    return get_average_values_df(tdf, step_size=step_size, field="Depth")


def enact_closure(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the entries over a given sample sum to unity

    Args:
        df_in (pd.DataFrame): dataframe of weight contributions 

    Returns:
        pd.DataFrame: dataframe of weight contributions normalised to one across each sample
    """
    return df_in.div(df_in.sum(axis=1), axis=0)


def merge_outputs(path1_in: Path|str, path2_in: Path|str, drop: list[str]=None, average_span: int=1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Typically you would use this to merge together a TSA SWIR, result and a TSA TIR result for example. 
    The dataframes are assumed to have come from the same TSG file after being extracted via the process_tsg_file method.
    You can drop columns by providing a list of column names and supply an averging span to reduce the data volume

    Args:
        df1_in (pd.DataFrame): dataframe 1, first column is the depth
        df2_in (pd.DataFrame): dataframe 2, first column is the depth
        drop (list[str], optional): list of column names to exclude and drop. Defaults to None.
        average_span (int, optional): The depth span over which to average. Defaults to 1.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The depth averaged merged dataframe and the original depth merged dataframe
    """
    if isinstance(path1_in, str):
        path1_in = Path(path1_in)
    if isinstance(path2_in, str):
        path2_in = Path(path2_in)
    
    # read in the csv files
    df1_in = pd.read_csv(path1_in)
    df2_in = pd.read_csv(path2_in)

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
    df_avg, _, _ = get_average_values_df(df, step_size=average_span, field="Depth (m)")
    df_avg.iloc[:, 1:] = df_avg.iloc[:, 1:].div(df_avg.iloc[:, 1:].sum(axis=1), axis=0)

    # enforce closure of what is being returned
    df[df.columns.tolist()[1:]] = enact_closure(df.iloc[:, 1:])
    df_avg[df_avg.columns.tolist()[1:]] = enact_closure(df_avg.iloc[:, 1:])

    # return the result
    return df_avg, df


def get_tsa_group_mineral_weight_columns(df_in: pd.DataFrame, search_space: tuple[str]) -> list[NDArray]:
    """_summary_

    Args:
        df_in (pd.DataFrame): A dataframe containing the TSG scalars data
        search_space (tuple[str]): The TSA or jclst names to look for so we can get the results for all levels e.g. min1, min2 etc

    Returns:
        list[NDArray]: A list of the actual names found in the incoming dataframe
    """
    # get the TSA information we need
    tsa_grp_columns = df_in.filter(regex=search_space[0]).columns.values
    tsa_min_columns = df_in.filter(regex=search_space[1]).columns.values
    tsa_wt_columns = df_in.filter(regex=search_space[2]).columns.values
    return [tsa_grp_columns, tsa_min_columns, tsa_wt_columns]


def get_tsa_results(df_in: pd.DataFrame, tsa_columns: list[str], tsa_weight_columns: list[str]) -> pd.DataFrame:
    """_summary_

    Args:
        df_in (pd.DataFrame): The TSG scalar dataframe
        tsa_columns (list[str]): A list of strings that correspond to the TSA, or jclst group or mineral entry names in the df
        tsa_weight_columns (list[str]): A list of strings that correspond to the TSA, or jclst group weigths for the tsa_columns

    Returns:
        pd.DataFrame: A dataframe where the index is the depth and each column is the name of the TSA, or jclst item and each entry is the weight
    """
    dfs = []
    df_use = df_in[['Depth (m)'] + list(tsa_columns) + list(tsa_weight_columns)].copy()
    df_use = df_use.drop_duplicates(subset='Depth (m)', keep='last')
    df_use = df_use.dropna(how='all', subset=df_use.columns[1:])
    for counter, value in enumerate(tsa_columns):
        dfs.append(df_use[['Depth (m)', value, tsa_weight_columns[counter]]].pivot(index='Depth (m)', columns=value,
                                                                                   values=tsa_weight_columns[counter]))

    df_concat = pd.concat(dfs, axis=1)
    df_out = df_concat.groupby(by=df_concat.columns, axis=1).sum().iloc[:,1:]
    return df_out


def process_tsa_spectral_regions(df: pd.DataFrame, path_out: Path, tsa_results: list[tuple]) -> bool:
    """
    Gather up all of the TSA, jclst data and process it into a dataframe then write the outputs as a csv

    Args:
        df (pd.DataFrame): Incoming dataframe containing the TSG scalar information (can also contains TSA and jclst data)
        path_out (Path): The path of where to write the output csv
        tsa_results (list[tuple]): A list of tuples containing the names of items to look for in the df (see process_tsg_file)

    Returns:
        bool: True if the tsa_results were found, False if not.
    """
    # process the various spectral regions
    for val in tsa_results:
        # get the TSA information we need/want
        tsa_grp_columns, tsa_min_columns, tsa_wt_columns = get_tsa_group_mineral_weight_columns(df, val)
        # make sure they are ordered e.g. 1, 2 & 3
        tsa_grp_columns.sort()
        tsa_min_columns.sort()
        tsa_wt_columns.sort()
        # make sure their is something to process first
        if tsa_grp_columns.size > 0:
            # get the group results
            df_group = get_tsa_results(df, tsa_grp_columns, tsa_wt_columns)
            # get the mineral results
            df_mineral = get_tsa_results(df, tsa_min_columns, tsa_wt_columns)
            # now define output file names write the data into an output CSV
            group_name = 'Group_' + tsa_grp_columns[0][-6:].strip() + '.csv'
            mineral_name = 'Mineral_' + tsa_grp_columns[0][-6:].strip() + '.csv'
            df_group.to_csv(str(path_out / group_name))
            df_mineral.to_csv(str(path_out / mineral_name))
            return True
        else:
            return False


def process_tsg_file(path_in: Path|str) -> None:
    """
    Do all of the stuff required to extract TSG data for later processing.
    This will be TSA data and jclst data primarily.

    This is the one to call!!

    Args:
        path (Path): The folder path (from a pathlib Path) to the TSG files location 
    """
    if isinstance(path_in, str):
        path_in = Path(path_in)

    all_tsg_files = [val for val in path_in.rglob("*.tsg")]
    for file in all_tsg_files:
        tsg_files = [file.name.replace(".tsg", ".bip"), file.name]
        #process_tsg_file(file.parent, files)

        # create the input files
        files = [str(path_in / tsg_files[0]), str(path_in / tsg_files[1])]
        
        # read in the TSG data.
        # if we are using Ben Chi's code here we need to see if its a VNIR/SWIR file or TIR file
        if 'tsg.tsg' in tsg_files[1]:
            data = parse_tsg.read_tsg_bip_pair(files[1], files[0], 'nir')
        else:
            data = parse_tsg.read_tsg_bip_pair(files[1], files[0], 'tir')

        # get the band header data e.g. scalars and other datasets like TSA etc
        df = data.scalars
        # apply the mask. I think it will also be 'On' or 'Off' in a TSG file. If its not it may break here
        df = df.iloc[data.scalars['Final Mask'].map({"On": True, "Off": False}).values, :]

        # define what we are going to search for and process. Yes I have added in the MIR even though its not there yet
        system_tsa_results = [('Grp. sTSAS', 'Min. sTSAS', 'Wt. sTSAS'),
                    ('Grp. sTSAV', 'Min. sTSAV', 'Wt. sTSAV'),
                    ('Grp. sTSAT', 'Min. sTSAT', 'Wt. sTSAT'),
                    ('Grp. sTSAM', 'Min. sTSAM', 'Wt. sTSAM')]
        process_tsa_spectral_regions(df, path_in, system_tsa_results)

        user_tsa_results = [('Grp. uTSAS', 'Min. uTSAS', 'Wt. uTSAS'),
                    ('Grp. uTSAV', 'Min. uTSAV', 'Wt. uTSAV'),
                    ('Grp. uTSAT', 'Min. uTSAT', 'Wt. uTSAT'),
                    ('Grp. uTSAM', 'Min. uTSAM', 'Wt. uTSAM')]
        process_tsa_spectral_regions(df, path_in, user_tsa_results)

        jclst_tsa_results = [('Grp. sjCLST', 'Min. sjCLST', 'Wt. sjCLST')]
        process_tsa_spectral_regions(df, path_in, jclst_tsa_results)

# example usage for extraction

# define the path where you are working. Use a string literal so you dont have to worry about / or \ stuff
path = Path(r'C:\Users\rod074\OneDrive - CSIRO\2021\mirewa\RTIO')
process_tsg_file(path)

# example usage for merging extracted TSG data 
# maybe you want to merge 2 of the output files. Maybe the system jclst result and a system TSA swir result
PATH1 = r"C:\Users\rod074\OneDrive - CSIRO\2022\minex-crc-op9\msdp01\Group_sjCLST.csv"
PATH2 = r"C:\Users\rod074\OneDrive - CSIRO\2022\minex-crc-op9\msdp01\Group_sTSAS.csv"
averaged_dataframe, nonaveraged_dataframe = merge_outputs(PATH1, PATH2)
