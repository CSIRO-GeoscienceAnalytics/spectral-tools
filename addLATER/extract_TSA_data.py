"""

"""
from spex.io.instruments import Tsg
from pathlib import Path
import pandas as pd


def get_TSA_group_mineral_weight_columns(df_in, search_space):
    """

    :param df_in:
    :type df_in:
    :param search_space:
    :type search_space:
    :return:
    :rtype:
    """
    # get the TSA information we need
    tsa_grp_columns = df_in.filter(regex=search_space[0]).columns.values
    tsa_min_columns = df_in.filter(regex=search_space[1]).columns.values
    tsa_wt_columns = df_in.filter(regex=search_space[2]).columns.values
    return [tsa_grp_columns, tsa_min_columns, tsa_wt_columns]


def get_TSA_results(df_in, tsa_columns, tsa_weight_columns):
    """

    :param df_in:
    :type df_in:
    :param tsa_columns:
    :type tsa_columns:
    :param tsa_weight_columns:
    :type tsa_weight_columns:
    :return:
    :rtype:
    """
    dfs = []
    df_use = df_in[['Depth (m)'] + list(tsa_columns) + list(tsa_weight_columns)].copy()
    df_use = df_use.drop_duplicates(subset='Depth (m)', keep='last')
    df_use = df_use.dropna(how='all', subset=df_use.columns[1:])
    for counter, value in enumerate(tsa_columns):
        dfs.append(df_use[['Depth (m)', value, tsa_weight_columns[counter]]].pivot(index='Depth (m)', columns=value,
                                                                                   values=tsa_weight_columns[counter]))

    df_concat = pd.concat(dfs, axis=1)
    df_out = df_concat.groupby(by=df_concat.columns, axis=1).sum()
    return df_out


def process_tsa_spectral_regions(df, path_out, tsa_results):
    """

    :param df:
    :type df:
    :param path_out:
    :type path_out:
    :param tsa_results:
    :type tsa_results:
    :return:
    :rtype:
    """
    # process the various spectral regions
    for val in tsa_results:
        # get the TSA information we need/want
        tsa_grp_columns, tsa_min_columns, tsa_wt_columns = get_TSA_group_mineral_weight_columns(df, val)
        # make sure they are ordered e.g. 1, 2 & 3
        tsa_grp_columns.sort()
        tsa_min_columns.sort()
        tsa_wt_columns.sort()
        # make sure their is something to process first
        if tsa_grp_columns.size > 0:
            # get the group results
            df_group = get_TSA_results(df, tsa_grp_columns, tsa_wt_columns)
            # get the mineral results
            df_mineral = get_TSA_results(df, tsa_min_columns, tsa_wt_columns)
            # now define output file names write the data into an output CSV
            group_name = 'Group_' + tsa_grp_columns[0][-6:].strip() + '.csv'
            mineral_name = 'Mineral_' + tsa_grp_columns[0][-6:].strip() + '.csv'
            df_group.to_csv(str(path_out / group_name))
            df_mineral.to_csv(str(path_out / mineral_name))


def process_tsg_file(path, tsg_files):
    """

    :param path:
    :type path:
    :param tsg_files:
    :type tsg_files:
    """
    # create the input files
    files = [str(path / tsg_files[0]), str(path / tsg_files[1])]
    # read in the TSG data
    data = Tsg(files)
    # get the band header data e.g. scalars and other datasets like TSA etc
    df = data.band_header_data
    # apply the mask
    df = df.iloc[data.mask, :]

    # define what we are going to search for and process. Yes I have added in the MIR even though its not there yet
    system_tsa_results = [('Grp. sTSAS', 'Min. sTSAS', 'Wt. sTSAS'),
                   ('Grp. sTSAV', 'Min. sTSAV', 'Wt. sTSAV'),
                   ('Grp. sTSAT', 'Min. sTSAT', 'Wt. sTSAT'),
                   ('Grp. sTSAM', 'Min. sTSAM', 'Wt. sTSAM')]
    process_tsa_spectral_regions(df, path, system_tsa_results)

    user_tsa_results = [('Grp. uTSAS', 'Min. uTSAS', 'Wt. uTSAS'),
                   ('Grp. uTSAV', 'Min. uTSAV', 'Wt. uTSAV'),
                   ('Grp. uTSAT', 'Min. uTSAT', 'Wt. uTSAT'),
                   ('Grp. uTSAM', 'Min. uTSAM', 'Wt. uTSAM')]
    process_tsa_spectral_regions(df, path, user_tsa_results)

    jclst_tsa_results = [('Grp. sjCLST', 'Min. sjCLST', 'Wt. sjCLST')]
    process_tsa_spectral_regions(df, path, jclst_tsa_results)


path = Path(r'C:\2021\mirewa\RTIO')
all_tsg_files = [val for val in path.rglob("*.tsg")]
for file in all_tsg_files:
    print(file.stem)
    files = [file.name.replace(".tsg", ".bip"), file.name]
    process_tsg_file(file.parent, files)
# files = ["287997_MSDP03_tsg.bip", "287997_MSDP03_tsg.tsg"]
# process_tsg_file(path, files)
