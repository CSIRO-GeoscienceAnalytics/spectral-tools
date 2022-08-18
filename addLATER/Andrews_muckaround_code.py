import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from Ancillary.ancillary import get_average_downhole_values, get_average_values_df, ternary_color
from Ancillary.spherical_projection import SphericalProjection
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from shapely.geometry import Point, Polygon
import Ancillary.ancillary as anc


########################################################################
# from spex.ext.chulls import get_absorption
# from spex.extraction.specex import SpectralExtraction
# path = Path(r'C:\2022\minex-crc-op9\data')
# files1 = [str(path/"18ABAD02_tsg.bip"), str(path/"18ABAD02_tsg.tsg")]
# files2 = [str(path/"18ABAD02_tsg_tir.bip"), str(path/"18ABAD02_tsg_tir.tsg")]
#
# sex1 = SpectralExtraction(files1, ordinate_inspection_range=[2050, 2480], do_hull=True, distance=4, prominence=0.005,
#                           height=0.006, width=3)
# sex2 = SpectralExtraction(files2, ordinate_inspection_range=[6000, 12000], do_hull=True, invert=True,
#                           distance=4, prominence=0.005,
#                           height=0.006, width=3)
#
# # get swir features
# feats1 = sex1.process_data()
# mask = sex1.instrument.mask
# # reorder so everything is in the lower triangle
# temp = np.array([[val[0], val[1], val[2]] if val[0] > val[1] else [val[1], val[0], val[2]] for val in feats1[mask,0,:3]])
# import hdbscan
# q = hdbscan.HDBSCAN(min_cluster_size=25)
# q.fit(temp[:,:2])
# df = pd.DataFrame(q.labels_, columns=["cluster"])
# df["Depth"] = sex1.instrument.band_header_data["Depth (m)"].values[mask]
# df["counter"] = df["Depth"] // 1
# x = df.groupby("counter")["Depth"].mean().values
# y = df.groupby(["counter"])["cluster"].agg(lambda x:x.value_counts().index[0]).values
# order = np.argsort(y)
# fig = px.histogram(x=x[order], y=y[order], barnorm="percent", color=y[order], nbins=x.shape[0])
# fig.show()
# order = np.argsort(q.labels_)
# values_in = feats1[mask, 0, :][order, :]
# colors_in = q.labels_[order].astype(str)
# fig = px.scatter(x=values_in[:,0], y=values_in[:,1], color=colors_in, color_discrete_sequence=px.colors.qualitative.Alphabet)
# fig.show()
# fig = px.scatter_3d(x=values_in[:, 0], y=values_in[:, 1], z=values_in[:, 2],color=colors_in, color_discrete_sequence=px.colors.qualitative.Alphabet)
# fig.show()
# spectra = []
# for val in np.unique(q.labels_):
#     z = np.where(q.labels_ == val)
#     spectra.append(1.0-get_absorption(sex1.ordinates, sex1.instrument.spectra[mask, :][z[0], :].mean(axis=0),0))
# spectra = np.array(spectra)
# spectral_df = pd.DataFrame(np.transpose(spectra), columns=np.unique(q.labels_).astype(str))
# spectral_df["wavelengths"] = sex1.instrument.ordinates
# fig = px.line(spectral_df, x="wavelengths", y=spectral_df.columns.tolist()[:-1], color_discrete_sequence=px.colors.qualitative.Alphabet, )
# fig. show()
#
# feats2 = sex2.process_data()
# bob=0




path = Path(r'\\fs1-cbr.nexus.csiro.au\{cesre-nvcl-hylogger}\work\common\MinExCRC_sandpit\Fortescue Group\18ABAD002_tsg')
file = "Group_sTSAS.csv"#"Group_sjCLST.csv"
file2 = "Group_sjCLST.csv"
df = pd.read_csv(path / file)
df2 = pd.read_csv(path / file2)
from merge_TSG_TSA_outputs import merge_outputs

df_avg, df_full = merge_outputs(df, df2, drop=["INVALID", "NOTAROK", "PHOSPHATE"])
fig = px.histogram(df_avg, x="Depth (m)", y=df_avg.iloc[:, 1:].columns, barmode='relative', barnorm="percent",
                   nbins=df_avg.shape[0], color_discrete_sequence=px.colors.qualitative.Alphabet)
fig.update_traces(marker_line_width=0)
fig.show()
bob=0
##################################################################################################################
# get all trhe MSDP paths
# all_msdp_paths = [val for val in
#                   Path(r'\\fs1-cbr.nexus.csiro.au\{cesre-nvcl-hylogger}\work\common\MinExCRC_sandpit').glob(
#                       "**/*MSDP*tsg.tsg")]
# # a list to hold the Group_sjCLST results
# dfs = []
# # loop over the paths
# for msdp_path in all_msdp_paths:
#     # define the file
#     file = msdp_path.parent / "Group_sjCLST.csv"
#     # read in the file
#     df = pd.read_csv(file)
#     # resample the file
#     df_out, _, _ = anc.get_average_values_df(df, field="Depth (m)")
#     df_out["Drillhole"] = msdp_path.stem
#     dfs.append(df_out)
# amalgamated = pd.concat(dfs)
# tm = TernaryMaker(internal_model='qapuv', colormap=px.colors.qualitative.Alphabet)
# closed = tm.enact_closure(amalgamated[["SILICA", "K-FELDSPAR", "PLAGIOCLASE"]])
# size = (25*amalgamated['KAOLIN']/amalgamated['KAOLIN'].max() + 2)
# fig = px.scatter_ternary(amalgamated, a="SILICA", b="K-FELDSPAR", c="PLAGIOCLASE", color=amalgamated["Drillhole"], size=size)
# fig.write_html("Amalgamated_MSDP_DrillHoles_size_is_Kaolin.html")
# fig.show()
#
# X, Y = tm.ternary_to_cartesian(closed["SILICA"], closed["K-FELDSPAR"])
# size = (25*amalgamated['KAOLIN']/amalgamated['KAOLIN'].max() + 2)
# fig = px.scatter_3d(x=X, y=Y, z=amalgamated["Depth (m)"], size=size, color_discrete_sequence=px.colors.qualitative.Dark24,
#                     color=amalgamated["Drillhole"], labels=["K-Feldspar", "Silica", "Depth (m)"])
# camera = dict(
#     eye=dict(x=0., y=0., z=-2),
#     up=dict(x=0, y=1, z=0)
# )
# fig.layout.scene.camera.eye = dict(x=0., y=0., z=-2)
# fig.layout.scene.camera.up = dict(x=0, y=1, z=0)
# fig.update_layout(scene_camera=camera, title='yo',
#                   scene=dict(
#                       xaxis_title='K-Feldspar',
#                       yaxis_title='Silica',
#                       zaxis_title='Depth (m)')
#                   )
# fig.write_html('amalgamated_toblerone.html')
# fig.show()
###################################################################################################################

# define the path and file we will be requiring
path = Path(r'C:\2022\minex-crc-op9\msdp01')
file = "Group_sjCLST.csv"#"Group_sjCLST.csv"
file2 = "Mineral_sjCLST.csv"
# path = Path(r'\\fs1-cbr.nexus.csiro.au\{cesre-nvcl-hylogger}\work\common\MinExCRC_sandpit\288779_MSDP13')
# file = "Group_sjCLST.csv"

# path = Path(r'\\fs1-cbr.nexus.csiro.au\{cesre-nvcl-hylogger}\work\common\MinExCRC_sandpit\287996_MSDP02')
# file = "Group_sjCLST.csv"
#
# path = Path(r'\\fs1-cbr.nexus.csiro.au\{cesre-nvcl-hylogger}\work\common\MinExCRC_sandpit\287998_MSDP04')
# file = "Group_sjCLST.csv"
#
# path = Path(r'\\fs1-cbr.nexus.csiro.au\{cesre-nvcl-hylogger}\work\common\MinExCRC_sandpit\288777_msdp11')
# file = "Group_sjCLST.csv"


# read in the csv
df = pd.read_csv(path / file)
bob=0
#fig = px.scatter(temp, x='Depth (m)', y="PYROXENE", size=temp["CHLORITE"]*8.0/temp["CHLORITE"].max() + 1.0, title="MSDP-01", color=temp["PYROXENE"]*8.0/temp["PYROXENE"].max() + 1.0)
#fig.write_html("chlorite_pyroxene_test.html")
#fig.show()

# fig = px.scatter_ternary(temp, a='SILICA', b="K-FELDSPAR", c="PLAGIOCLASE", size=temp["CHLORITE"]*10.0/temp["CHLORITE"].max() + 1.0,
#                          color="PYROXENE", title="MSDP-01: size=chlorite, color=pyroxene", color_continuous_scale=px.colors.sequential.Jet)
# fig.write_html("ternary_chlorite_pyroxene_test.html")
# fig.show()
#
# fig = px.scatter_ternary(temp, a='SILICA', b="K-FELDSPAR", c="PLAGIOCLASE", size=temp["PYROXENE"]*10.0/temp["PYROXENE"].max() + 1.0,
#                          color="CHLORITE", title="MSDP-01: size=pyroxene, color=chlorite", color_continuous_scale=px.colors.sequential.Jet)
# fig.write_html("ternary_pyroxene_chlorite_test.html")
# fig.show()

# fig = px.bar(temp, x="Depth (m)", y="CHLORITE", color="PYROXENE")
# fig.show()


#df2 = pd.read_csv(path / file2)
tm = TernaryMaker(internal_model='qapuv', colormap=px.colors.qualitative.Alphabet)
ff = tm.downhole_barplot(df, step_field=['Depth (m)', 'SILICA', 'K-FELDSPAR', 'PLAGIOCLASE'], step_size=1)
ff.show()

f = tm.overlay_on_ternary(df, step_field=['Depth (m)', 'SILICA', 'K-FELDSPAR', 'PLAGIOCLASE'], step_size=1)#, ternary_size='CHLORITE')
f.show()

####3D Ternary
df_out, _, _ = anc.get_average_values_df(df, step_size=1, field="Depth (m)")
temp =tm.get_ternary_designation(df_out[["SILICA", "K-FELDSPAR", "PLAGIOCLASE"]])
X, Y = tm.ternary_to_cartesian(temp["SILICA"], temp["K-FELDSPAR"])
size = (25*df_out['PYROXENE']/df_out['PYROXENE'].max() + 2)
fig = px.scatter_3d(x=X, y=Y, z=df_out["CHLORITE"], size=size, color_discrete_sequence=px.colors.qualitative.Alphabet,
                    color=temp["Ternary"])
camera = dict(
    eye=dict(x=0., y=0., z=-2),
    up=dict(x=0, y=1, z=0)
)
fig.layout.scene.camera.eye = dict(x=0., y=0., z=-2)
fig.layout.scene.camera.up = dict(x=0, y=1, z=0)
fig.update_layout(scene_camera=camera, title='yo')
fig.write_html('just_for_fun.html')
fig.show()


#
# cal98 = ['albite', 'actinolite', 'adularia', 'alunite', 'andalusite', 'biotite', 'carbonate', 'chlorite',
#  'chabazite', 'chalcedony', 'chlorite-smectite', 'corundum', 'clinopyroxene',
#  'cristobalite', 'calcite', 'dolomite', 'dickite', 'diaspore', 'epidote', 'feldspar', 'garnet',
#  'halloysite', 'heulandite', 'illite', 'illite-smectite', 'kaolinite', 'laumonite', 'magnetite',
#  'mordenite', 'natrolite', 'opaline silica', 'pyrophyllite', 'quartz', 'sericite', 'siderite', 'smectite', 'stibnite',
#  'tremolite', 'tridymite', 'vesuvianite', 'wairakite', 'wollastonite', 'zeolite']
# tsa_minerals = pd.read_csv("tsa_minerals.csv")
# result = [df_avg.columns.str.contains(val, case=False).max() for val in cal98]
# what = np.array(cal98)[result]
bob=0