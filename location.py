from flask import Blueprint, render_template,request
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
import haversine as hs


import plotly.io as pio
pio.renderers.default = 'browser'

location = Blueprint("location",__name__)

@location.route("/location")

def loc():
    

    india_states = json.load(open("states_india.geojson", "r"))

    state_id_map = {}
    for feature in india_states["features"]:
        feature["id"] = feature["properties"]["state_code"]
        state_id_map[feature["properties"]["st_nm"]] = feature["id"]
        
    df = pd.read_csv("india_census.csv")
    df["Density"] = df["Density[a]"].apply(lambda x: int(x.split("/")[0].replace(",", "")))
    df["id"] = df["State or union territory"].apply(lambda x: state_id_map[x])

    df.head()

    df["Density"].plot()

    df["DensityScale"] = np.log10(df["Density"])

    df["DensityScale"].plot()

    # fig1 = px.choropleth(
    #     df,
    #     locations="id",
    #     geojson=india_states,
    #     color="DensityScale",
    #     hover_name="State or union territory",
    #     hover_data=["Density"],
    #     title="India Population Density",
    # )

    # fig1.update_geos(fitbounds="locations", visible=False)
    # graph1JSON=json.dumps(fig1,cls = plotly.utils.PlotlyJSONEncoder)
    
    
    ##total population
    fig1 = px.choropleth_mapbox(
        df,
        locations="id",
        geojson=india_states,
        color="Population",
        hover_name="State or union territory",
        hover_data=["Population"],
        title="India Population per state",
        mapbox_style="carto-positron",
        center={"lat": 24, "lon": 78},
        zoom=3,
        opacity=0.5,
    )
    graph1JSON=json.dumps(fig1,cls = plotly.utils.PlotlyJSONEncoder)
    
    
    
    ##population density
    fig2 = px.choropleth_mapbox(
        df,
        locations="id",
        geojson=india_states,
        color="Density",
        hover_name="State or union territory",
        hover_data=["Density"],
        title="India Population Density",
        mapbox_style="carto-positron",
        center={"lat": 24, "lon": 78},
        zoom=3,
        opacity=0.5,
    )
    graph2JSON=json.dumps(fig2,cls = plotly.utils.PlotlyJSONEncoder)
    
    
    
    ##sex ratio
    df["Sex ratio"] = df["Sex ratio"] - 1000
    
    fig3 = px.choropleth_mapbox(
        df,
        locations="id",
        geojson=india_states,
        color="Sex ratio",
        hover_name="State or union territory",
        hover_data=["Sex ratio"],
        title="India sex ratio",
        mapbox_style="carto-positron",
        center={"lat": 24, "lon": 78},
        zoom=3,
        opacity=0.5,
    )
    graph3JSON=json.dumps(fig3,cls = plotly.utils.PlotlyJSONEncoder)    
        
        
    ##Road density
    fig4 = px.choropleth_mapbox(
        df,
        locations="id",
        geojson=india_states,
        color="road_den",
        hover_name="State or union territory",
        hover_data=["road_den"],
        title="Density of roads",
        mapbox_style="carto-positron",
        center={"lat": 24, "lon": 78},
        zoom=3,
        opacity=0.5,
    )
    graph4JSON=json.dumps(fig4,cls = plotly.utils.PlotlyJSONEncoder)    
        
        
    return render_template("location.html",graph1JSON=graph1JSON,graph2JSON=graph2JSON,graph3JSON=graph3JSON,
                           graph4JSON=graph4JSON)


@location.route("/location", methods={'POST'})

def dis():


    # lat_f=float(input("input latitude for factory: "))
    # lon_f=float(input("input longitude for factory: "))
    # lat_m=float(input("input latitude for market: "))
    # lon_m=float(input("input longitude for market: "))
    lat_f=request.form['FacLat']
    lon_f=request.form['FacLot']
    lat_m=request.form['SalLat']
    lon_m=request.form['SalLot']


    factory_location=(float(lat_f),float(lon_f))
    market_location=(float(lat_m),float(lon_m))
    return render_template('location.html', variable=hs.haversine(factory_location,market_location))
    #print(hs.haversine(factory_location,market_location))
