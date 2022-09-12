"""
02/05/2022
Andrew Rodger, CSIRO, Mineral Resources

A fairly simple class for creating ternary diagrams and classing TSG data (jCLST at this stage) into the relevant
QAP type classes.
Internal Ternary Diagrams are:
1) The upper triangle of the Plutonic QAPF diagram https://upload.wikimedia.org/wikipedia/commons/a/af/Intrusive_big.png
2) The ultramafic Ol Opx Cpx diagram https://www.mindat.org/photo-471844.html
3) The ultramafic Ol Px Hbl diagram https://www.mindat.org/photo-471844.html

I will add more as they are made known to me.

The inputs for the class if you are wanting to see what falls where are coming so far from extracted mineral group data
from TSG. For the QAP diagram I have been using Silica, K-Feldpsar and Plagioclase as the QA and P. All of this data has
come from the jCLST estimation at the group level.
"""
import math

import matplotlib.path as mplPath
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TernaryMaker:
    """

    """
    def __init__(self, use_internal=True, internal_model='qapup', colormap=None):
        if colormap:
            self.colormap = colormap
        else:
            self.colormap = px.colors.qualitative.Dark24

        self.cartesian_polygons = None
        self.points = None
        self.names = None
        self.ternary_polygon_vertices = None
        self.axis_names = None
        self.fig1 = None
        self.traces = None

        if use_internal:
            self.use_internal = True
            self.internal_model = internal_model
            if self.internal_model == 'qapup':
                self.set_qap_upper_plutonic_data()
            if self.internal_model == 'qapuv':
                self.set_qap_upper_volcanic_data()
            if self.internal_model == 'uooc':
                self.set_ultramafic_Ol_Opx_Cpx()
            if self.internal_model == 'uoph':
                self.set_ultramafic_Ol_Px_Hbl()
        else:
            self.use_internal = False
            self.internal_model = internal_model

    def set_qap_upper_plutonic_data(self):
        self.internal_model == 'qapup'
        # define the names
        self.names = ['Quartzolite', 'Quartz-Rich Granitoid', 'Alkali-Feldspar Granite', 'Syenogranite', 'Monzogranite',
                 'Granodiorite', 'Tonalite',
                 'Quartz-Bearing Alkali-Feldspar Syenite', 'Quartz-Bearing Syenite', 'Quartz-Monzonite',
                 'Quartz-Monzodiorite / Quartz-Monzogabbro', 'Quartz-Diorite / Quartz-Gabbro',
                 'Alkali-Feldspar Syenite',
                 'Syenite', 'Monzonite', 'Monzodiorite / Monzogabbro', 'Diorite / Gabbro / Anorthosite']
        # define each regions coordinate values
        self.ternary_polygon_vertices = [
            [[1, 0.9, 0.9], [0, 0.1, 0.0], [0, 0.0, 0.1]],
            [[0.9, 0.6, 0.6, 0.9], [0.1, 0.4, 0.0, 0.0], [0.0, 0.0, 0.4, 0.1]],
            [[0.6, 0.2, 0.2, 0.6], [0.4, 0.8, 0.72, 0.36], [0, 0, 0.08, 0.04]],
            [[0.6, 0.2, 0.2, 0.6], [0.36, 0.72, 0.52, 0.26], [0.04, 0.08, 0.28, 0.14]],
            [[0.6, 0.2, 0.2, 0.6], [0.26, 0.52, 0.28, 0.14], [0.14, 0.28, 0.52, 0.26]],
            [[0.6, 0.2, 0.2, 0.6], [0.14, 0.28, 0.08, 0.04], [0.26, 0.52, 0.72, 0.36]],
            [[0.6, 0.2, 0.2, 0.6], [0.04, 0.08, 0.0, 0.0], [0.36, 0.72, 0.8, 0.4]],
            [[0.2, 0.05, 0.05, 0.2], [0.8, 0.95, 0.855, 0.72], [0.0, 0.0, 0.095, 0.08]],
            [[0.2, 0.05, 0.05, 0.2], [0.72, 0.855, 0.6175, 0.52], [0.08, 0.095, 0.3325, 0.28]],
            [[0.2, 0.05, 0.05, 0.2], [0.52, 0.6175, 0.3325, 0.28], [0.28, 0.3325, 0.6175, 0.52]],
            [[0.2, 0.05, 0.05, 0.2], [0.28, 0.3325, 0.095, 0.08], [0.52, 0.6175, 0.855, 0.72]],
            [[0.2, 0.05, 0.05, 0.2], [0.08, 0.095, 0.0, 0.0], [0.72, 0.855, 0.95, 0.8]],
            [[0.05, 0.0, 0.0, 0.05], [0.95, 1, 0.9, 0.855], [0, 0, 0.1, 0.095]],
            [[0.05, 0.0, 0.0, 0.05], [0.855, 0.9, 0.65, 0.6175], [0.095, 0.1, 0.35, 0.3325]],
            [[0.05, 0.0, 0.0, 0.05], [0.6175, 0.65, 0.35, 0.3325], [0.3325, 0.35, 0.65, 0.6175]],
            [[0.05, 0.0, 0.0, 0.05], [0.3325, 0.35, 0.1, 0.095], [0.6175, 0.65, 0.9, 0.855]],
            [[0.05, 0.0, 0.0, 0.05], [0.095, 0.1, 0, 0], [0.855, 0.9, 1, 0.95]],
            ]
        self.axis_names = ["Q", "A", "P"]
        self.ternary_polygon_vertices_to_cartesian_polygons()
        self.make_ternary()

    def set_qap_upper_volcanic_data(self):
        """

        """
        self.internal_model == 'qapuv'
        # define the names
        self.names = ['Undefined', 'Alkali - Feldspar Rhyolite', 'Rhyolite', 'Dacite', 'Plagidacite',
                 'Quartz - Bearing Alkali - Feldspar Trachyte', 'Quartz - Trachyte', 'Quartz-Latite', 'Quartz - Bearing Andesite',
                      'Quartz - Bearing Basalt',
                      'Alkali - Feldspar Trachyte', 'Trachyte','Latite', 'Andesite', 'Basalt']
        # define each regions coordinate values
        self.ternary_polygon_vertices = [
            [[1, 0.6, 0.6], [0, 0.4, 0.0], [0, 0.0, 0.4]],
            [[0.6, 0.2, 0.2, 0.6], [0.4, 0.8, 0.72, 0.36], [0, 0, 0.08, 0.04]],
            [[0.6, 0.2, 0.2, 0.6], [0.36, 0.72, 0.28, 0.14], [0.04, 0.08, 0.52, 0.26]],
            [[0.6, 0.2, 0.2, 0.6], [0.14, 0.28, 0.08, 0.04], [0.26, 0.52, 0.72, 0.36]],
            [[0.6, 0.2, 0.2, 0.6], [0.04, 0.08, 0.0, 0.0], [0.36, 0.72, 0.8, 0.4]],
            [[0.2, 0.05, 0.05, 0.2], [0.8, 0.95, 0.855, 0.72], [0.0, 0.0, 0.095, 0.08]],
            [[0.2, 0.05, 0.05, 0.2], [0.72, 0.855, 0.6175, 0.52], [0.08, 0.095, 0.3325, 0.28]],
            [[0.2, 0.05, 0.05, 0.2], [0.52, 0.6175, 0.3325, 0.28], [0.28, 0.3325, 0.6175, 0.52]],
            [[0.2, 0.05, 0.05, 0.2], [0.28, 0.3325, 0.095, 0.08], [0.52, 0.6175, 0.855, 0.72]],
            [[0.2, 0.05, 0.05, 0.2], [0.08, 0.095, 0.0, 0.0], [0.72, 0.855, 0.95, 0.8]],
            [[0.05, 0.0, 0.0, 0.05], [0.95, 1, 0.9, 0.855], [0, 0, 0.1, 0.095]],
            [[0.05, 0.0, 0.0, 0.05], [0.855, 0.9, 0.65, 0.6175], [0.095, 0.1, 0.35, 0.3325]],
            [[0.05, 0.0, 0.0, 0.05], [0.6175, 0.65, 0.35, 0.3325], [0.3325, 0.35, 0.65, 0.6175]],
            [[0.05, 0.0, 0.0, 0.05], [0.3325, 0.35, 0.1, 0.095], [0.6175, 0.65, 0.9, 0.855]],
            [[0.05, 0.0, 0.0, 0.05], [0.095, 0.1, 0, 0], [0.855, 0.9, 1, 0.95]],
            ]
        self.axis_names = ["Q", "A", "P"]
        self.ternary_polygon_vertices_to_cartesian_polygons()
        self.make_ternary()

    def set_ultramafic_Ol_Opx_Cpx(self):

        self.internal_model == 'uooc'
        self.names = ['Dunite',
                 'Harzburgite', 'Lherzolite', 'Wehrlite',
                 'Olivine Orthopyroxenite', 'Olivine Websterite', 'Olivine Clinopyroxenite',
                 'Orthopyroxenite', 'Websterite', 'Clinopyroxenite']
        # define each regions coordinate values
        self.ternary_polygon_vertices = [[[1, 0.9, 0.9], [0, 0.1, 0], [0, 0, 0.1]],
                    [[0.9, 0.4, 0.4, 0.9], [0.1, 0.6, 0.55, 0.05], [0, 0.0, 0.05, 0.05]],
                    [[0.9, 0.4, 0.4], [0.05, 0.55, 0.05], [0.05, 0.05, 0.55]],
                    [[0.9, 0.4, 0.4, 0.9], [0.05, 0.05, 0.0, 0.0], [0.05, 0.55, 0.6, 0.1]],
                    [[0.4, 0.1, 0.05, 0.4], [0.6, 0.9, 0.9, 0.55], [0.0, 0.0, 0.05, 0.05]],
                    [[0.4, 0.05, 0.05, 0.4], [0.55, 0.9, 0.05, 0.05], [0.05, 0.05, 0.9, 0.55]],
                    [[0.4, 0.05, 0.1, 0.4], [0.05, 0.05, 0.0, 0.0], [0.55, 0.9, 0.9, 0.6]],
                    [[0.1, 0.0, 0.0], [0.9, 1.0, 0.9], [0.0, 0.0, 0.1]],
                    [[0.05, 0.0, 0.0, 0.05], [0.9, 0.9, 0.1, 0.05], [0.05, 0.1, 0.9, 0.9]],
                    [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.9, 0.9, 1.0]]
                    ]
        self.axis_names = ["Ol", "Opx", "Cpx"]
        self.ternary_polygon_vertices_to_cartesian_polygons()
        self.make_ternary()

    def set_ultramafic_Ol_Px_Hbl(self):
        self.internal_model == 'uoph'
        self.names = ['Dunite',
                 'Pyroxene Peridotite', 'Pyroxene Hornblende Peridotite', 'Hornblende Peridotite',
                 'Olivine Pyroxenite', 'Olivine Hornblende Pyroxenite', 'Olivine Pyroxene Hornblendite',
                 'Olivine Hornblendite',
                 'Pyroxenite', 'Hornblende Pyroxinite', 'Pyroxene Hornblendite', 'Hornblendite']
        # define each regions coordinate values
        self.ternary_polygon_vertices = [[[1, 0.9, 0.9], [0, 0.1, 0], [0, 0, 0.1]],
                    [[0.9, 0.4, 0.4, 0.9], [0.1, 0.6, 0.55, 0.05], [0, 0.0, 0.05, 0.05]],
                    [[0.9, 0.4, 0.4], [0.05, 0.55, 0.05], [0.05, 0.05, 0.55]],
                    [[0.9, 0.4, 0.4, 0.9], [0.05, 0.05, 0.0, 0.0], [0.05, 0.55, 0.6, 0.1]],
                    [[0.4, 0.1, 0.05, 0.4], [0.6, 0.9, 0.9, 0.55], [0.0, 0.0, 0.05, 0.05]],
                    [[0.4, 0.05, 0.05, 0.4], [0.55, 0.9, 0.475, 0.3], [0.05, 0.05, 0.475, 0.3]],
                    [[0.4, 0.05, 0.05, 0.4], [0.3, 0.475, 0.05, 0.05], [0.3, 0.475, 0.9, 0.55]],
                    [[0.4, 0.05, 0.1, 0.4], [0.05, 0.05, 0.0, 0.0], [0.55, 0.9, 0.9, 0.6]],
                    [[0.1, 0.0, 0.0], [0.9, 1.0, 0.9], [0.0, 0.0, 0.1]],
                    [[0.05, 0.0, 0.0, 0.05], [0.9, 0.9, 0.5, 0.475], [0.05, 0.1, 0.5, 0.475]],
                    [[0.05, 0.0, 0.0, 0.05], [0.475, 0.5, 0.1, 0.05], [0.475, 0.5, 0.9, 0.9]],
                    [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.9, 0.9, 1.0]]
                    ]
        self.axis_names = ["Ol", "Px", "Hbl"]
        self.ternary_polygon_vertices_to_cartesian_polygons()
        self.make_ternary()

    def ternary_to_cartesian(self, A, B):
        Y = A * 0.5 * math.tan(60 * math.pi / 180.)
        X = A * math.tan(30 * math.pi / 180.) + B
        return X, Y

    def enact_closure(self, df_in):
        return df_in.div(df_in.sum(axis=1), axis=0)

    def ternary_polygon_vertices_to_cartesian_polygons(self):
        polygons = []
        for val in enumerate(self.ternary_polygon_vertices):
            # convert the QAP values to cartesian coordinates : used later for seeing whats in each QAP polygon
            x, y = self.ternary_to_cartesian(np.array(val[0]), np.array(val[1]))
            polygons.append(np.concatenate((x[:, None], y[:, None]), axis=1))
        self.cartesian_polygons = polygons

    def make_ternary(self):
        # define the names
        names = self.names
        # define each regions coordinate values
        polys = self.ternary_polygon_vertices
        # create an empty list to hold our traces
        traces = []
        # cycle through the q_points and create our traces
        for index, val in enumerate(polys):
            # append the trace
            traces.append(go.Scatterternary(a=val[0], b=val[1], c=val[2], mode='lines', line=dict(color='black', width=0.5),
                                            fill='toself', fillcolor=self.colormap[index],
                                            name=names[index], hovertext=[names[index]]))

        # add the traces to the figure
        fig = go.Figure(data=traces)
        # update the axis labels on the ternary
        fig.update_layout({
            'ternary':
                {
                    'aaxis': {'title': self.axis_names[0]},
                    'baxis': {'title': self.axis_names[1]},
                    'caxis': {'title': self.axis_names[2]}
                },
        })
        self.fig1 = fig
        self.traces = traces
        return self.fig1, self.traces

    def get_ternary_designation(self, df_in):


        # enforce closure
        temp = self.enact_closure(df_in)

        # ternary to cartesian
        X, Y = self.ternary_to_cartesian(temp.iloc[:, 0], temp.iloc[:, 1])

        # add a new column to store the ternary designation
        temp.loc[:, "Ternary"] = "Unclassed"

        # create a point array for use with the shapely library
        points = [(val[0], val[1]) for val in zip(X, Y)]
        self.points = points
        # see what points are in what ternary polygon
        for index, polygon in enumerate(self.cartesian_polygons):
            poly = mplPath.Path(polygon)
            # if a +ve radius is used here it leaves a number of points unclassed
            # if its -ve is doesnt. Not sure why. A google search also shows that others dont really know why either
            # including the folks who made it :)
            test_points = poly.contains_points(points, radius=-0.0001)
            temp["Ternary"].loc[test_points] = self.names[index]

        # Alternative way to do it. Produces some misses though
        # P = [Point(val) for val in points]
        # pts = pd.Series(P)
        # polys = [Polygon(val) for val in self.cartesian_polygons]
        # for index, poly in enumerate(polys):
        #     test_points = pts.map(lambda p: poly.intersects(p))
        #     temp['Ternary'].loc[test_points] = self.names[index]

        return temp

    def downhole_barplot(self, df_in, step_field=None, step_size=None):

        # TODO fix up the whole resample thing. Its a bit of mess how the info comes in. Maybe I need a dict?

        if step_field:
            df_out, _, _ = self.get_average_values_df(df_in[step_field], step_size=step_size, field=step_field[0])
            temp = self.get_ternary_designation(df_out.iloc[:, 1:])
        else:
            df_out, _, _ = self.get_average_values_df(df_in.iloc[:, :3], step_size=step_size)
            temp = self.get_ternary_designation(df_out)

        # add the extra names to the names list
        names2 = self.names + ['Unclassed']

        # make a downhole bar plot
        fig = go.Figure()
        for index, qap in enumerate(names2):
            if step_field:
                x = df_out[step_field[0]][temp["Ternary"] == qap]
            else:
                x = temp[temp["Ternary"] == qap].index.values
            y = [100] * x.shape[0]
            fig.add_trace(go.Bar(x=x, y=y, marker_color=self.colormap[index], name=qap))
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0, xaxis_showgrid=False, yaxis_showgrid=False)
        return fig

    def overlay_on_ternary(self, df_in, step_size=1, step_field=None, ternary_size=None):
        # TODO what if you dont want to average because you have already done it?
        if step_field:
            df_out, _, _ = self.get_average_values_df(df_in, step_size=step_size, field=step_field[0])
            temp = self.get_ternary_designation(df_out.loc[:, step_field[1:]])
        else:
            df_out, _, _ = self.get_average_values_df(df_in.iloc[:, :3], step_size=step_size)
            temp = self.get_ternary_designation(df_out)

        # add the extra names to the names list
        names2 = self.names + ['Unclassed']

        # side by side QAP
        for trace in self.traces:
            trace.showlegend = False
        f = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, specs=[[{"type": "ternary"}, {"type": "ternary"}]])
        # add ternary polygons
        trace_length = self.traces.__len__()
        f.add_traces(self.traces, rows=[1] * trace_length, cols=[1] * trace_length)
        # add data points to ternary
        hovertemplate=None
        if self.use_internal:
            if self.internal_model == 'qapup':
                hovertemplate = 'Q: %{a}' + '<br>A: %{b}</br>' + 'P: %{c}' + '<br>%{text}</br>'
            if self.internal_model == 'uooc':
                hovertemplate = 'Ol: %{a}' + '<br>Opx: %{b}</br>' + 'Cpx: %{c}' + '<br>%{text}</br>'
            if self.internal_model == 'uoph':
                hovertemplate = 'Ol: %{a}' + '<br>Px: %{b}</br>' + 'Hbl: %{c}' + '<br>%{text}</br>'

        f.add_trace(go.Scatterternary(a=temp.iloc[:, 0], b=temp.iloc[:, 1], c=temp.iloc[:, 2], mode='markers',
                                      marker=dict(color='white', symbol='circle',
                                                  line=dict(width=2, color='DarkSlateGrey')),
                                      showlegend=False, hovertemplate=hovertemplate,
                                      text=['{}'.format(val) for val in temp["Ternary"]]), row=1, col=1)

        # add second ternary coloured by QAP polygon
        if ternary_size:
            size_info = (15*df_out[ternary_size]/df_out[ternary_size].max() + 5)
        else:
            size_info = None

        for index, qap in enumerate(names2):

            if ternary_size:
                size_measure = size_info[temp['Ternary'] == qap]
            else:
                size_measure = None

            f.add_trace(
                go.Scatterternary(a=temp.iloc[:, 0][temp["Ternary"] == qap], b=temp.iloc[:, 1][temp["Ternary"] == qap],
                                  c=temp.iloc[:, 2][temp["Ternary"] == qap],
                                  mode='markers',
                                  marker=dict(color=self.colormap[index], symbol='circle',
                                              size=size_measure), name=qap,
                                  hovertemplate=None),
                row=1, col=2)
        f.update_layout({
            'ternary':
                {
                    'aaxis': {'title': self.axis_names[0]},
                    'baxis': {'title': self.axis_names[1]},
                    'caxis': {'title': self.axis_names[2]}
                },
            'ternary2':
                {
                    'aaxis': {'title': self.axis_names[0]},
                    'baxis': {'title': self.axis_names[1]},
                    'caxis': {'title': self.axis_names[2]}
                },
        }, title_text="")
        return f

    def get_average_values_df(self, df_in, step_size=1, field=None):
        """

        :param df_in: incoming dataframe
        :type df_in: pandas dataframe
        :param step_size: the integer divisor
        :type step_size: integer
        :param field: the column in the dataframe to use for the integer division
        :type field: string
        :return: a new dataframe that is averaged according to the integer divisor section_start values, section_stop values
        :rtype: pandas dataframe, pandas series, pandas series
        """
        import pandas as pd

        def from_to(df_series, step):
            """

            :param df_series: panda series of values to find start and stop values from
            :type df_series: pandas series
            :param step: integer divisor step
            :type step: integer
            :return: start and stop values of each section
            :rtype: pandas series, pandas series
            """
            start = df_series[np.unique(df_series // step, return_index=True)[1]]
            stop = pd.concat([df_series[np.unique(df_series // step, return_index=True)[1][1:] - 1],
                              pd.Series(df_series.values[-1])])
            return start, stop

        if field is None:
            section_start, section_stop = from_to(pd.Series(df_in.index), step_size)
            return df_in.groupby(df_in.index // step_size).mean(), section_start, section_stop
        else:
            section_start, section_stop = from_to(df_in[field], step_size)
            return df_in.groupby(df_in[field] // step_size).mean(), section_start, section_stop


