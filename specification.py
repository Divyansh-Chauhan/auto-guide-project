from flask import Blueprint, render_template
import pandas as pd     
import numpy as np
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import json
import plotly
import seaborn as sns


specification = Blueprint("specification",__name__)

@specification.route("/specification")

def spec():

                                    ###DATA CLEANING CODE TO SAVE INTO A NEW CSV#######

    # df = pd.read_csv('data.csv')


    # #Car_Name variable created to access make + model at once
    # df['Car_Name'] = df.Make+ ' ' +df.Model


    # #Dataset cleaning
    # necessary_features = ['Make','Model','Car_Name','Variant','Body_Type','Fuel_Type','Fuel_System','Type','Drivetrain',
    #                       'Ex-Showroom_Price','Displacement','Cylinders','ARAI_Certified_Mileage','Power','Torque',
    #                       'Fuel_Tank_Capacity','Height','Length','Width','Doors','Seating_Capacity','Wheelbase',
    #                       'Number_of_Airbags']

    # df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].apply(str).str.replace( 'Rs. ' , '' ,regex = False)
    # df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].apply(str).str.replace( ',' , '' ,regex = False)
    # df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].apply(str).astype(int)
    # df = df[necessary_features]
    # df = df[~df.ARAI_Certified_Mileage.isnull()]
    # df = df[~df.Make.isnull()]
    # df = df[~df.Width.isnull()]
    # df = df[~df.Cylinders.isnull()]
    # df = df[~df.Wheelbase.isnull()]
    # df = df[~df['Fuel_Tank_Capacity'].isnull()]
    # df = df[~df['Seating_Capacity'].isnull()]
    # df = df[~df['Torque'].isnull()]
    # df['Height'] = df['Height'].apply(str).str.replace(' mm','',regex=False).astype(float)
    # df['Length'] = df['Length'].apply(str).str.replace(' mm','',regex=False).astype(float)
    # df['Width'] = df['Width'].apply(str).str.replace(' mm','',regex=False).astype(float)
    # df['Wheelbase'] = df['Wheelbase'].apply(str).str.replace(' mm','',regex=False).astype(float)#apply(str) is used for sorting some error
    # df['Fuel_Tank_Capacity'] = df['Fuel_Tank_Capacity'].apply(str).str.replace(' litres','',regex=False).astype(float)
    # df['Displacement'] = df['Displacement'].apply(str).str.replace(' cc','',regex=False)
    # df.loc[df.ARAI_Certified_Mileage == '9.8-10.0 km/litre','ARAI_Certified_Mileage'] = '10'
    # df.loc[df.ARAI_Certified_Mileage == '10kmpl km/litre','ARAI_Certified_Mileage'] = '10'
    # df['ARAI_Certified_Mileage'] = df['ARAI_Certified_Mileage'].apply(str).str.replace(' km/litre','',regex=False).astype(float)
    # df.Number_of_Airbags.fillna(0,inplace= True)

    # #on-road price difference is very high in India, so that column will be considered

    # #tax is from 3-20% so 20% was taken to cover max attributes
    # df['GST_price'] = df['Ex-Showroom_Price'] * 0.020  
    # df.GST_price = df.GST_price.astype(int)
    # df['On_road-price'] = df['GST_price'] + df['Ex-Showroom_Price']
    # df.drop(columns = 'Ex-Showroom_Price', inplace= True )
    # df.drop(columns = 'GST_price', inplace= True )
    # df.drop(columns = 'Fuel_System', inplace= True )
    # HP = df.Power.str.extract(r'(\d{1,4}).*').astype(int) * 0.98632
    # HP = HP.apply(lambda x: round(x,2))
    # TQ = df.Torque.str.extract(r'(\d{1,4}).*').astype(int)
    # TQ = TQ.apply(lambda x: round(x,2))
    # df.Torque = TQ
    # df.Power = HP
    # df.Doors = df.Doors.astype(int)
    # df.Seating_Capacity = df.Seating_Capacity.astype(int)
    # df.Number_of_Airbags = df.Number_of_Airbags.astype(int)
    # df.Displacement = df.Displacement.astype(int)
    # df.Cylinders = df.Cylinders.astype(int)
    # df.columns = ['Manufacturer', 'Model', 'Car_Name', 'Variant', 'Body_Type', 'Fuel_Type','Gear_Type', 'Drivetrain', 'Displacement', 'Cylinders',
    #               'Mileage', 'Power', 'Torque', 'Fuel_tank','Height', 'Length', 'Width', 'Doors', 'Seats', 'Wheelbase','Airbags', 'On_road_price']
    # df = df.dropna()


    # df.to_csv('cleaned-data.csv')


                    ######################################     END     ###################################

    df = pd.read_csv('cleaned-data.csv')

    #pie chart for top 10 companies
    make_names = df.Manufacturer.value_counts().index[:10]
    make_values = df.Manufacturer.value_counts().values[:10]

    fig1 = px.pie(df, values=make_values, names=make_names, title='Top 10 manufacteres with most models in the market',hole = 0.20)
    graph1JSON=json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)



    #Box plot of onroad price
    fig2 = px.box(df, x="On_road_price", points="all")
    graph2JSON=json.dumps(fig2,cls=plotly.utils.PlotlyJSONEncoder)



    #pie chart for body types
    fig3 = px.pie(df,names=df.Body_Type.value_counts().index,values=df.Body_Type.value_counts(),
    color_discrete_sequence=px.colors.sequential.RdBu, width=600, height=600)
    fig3.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
    graph3JSON=json.dumps(fig3,cls=plotly.utils.PlotlyJSONEncoder)



    #Box plot of onroad price and body type
    fig4 = px.box(df, x="On_road_price",y="Body_Type")
    graph4JSON=json.dumps(fig4,cls=plotly.utils.PlotlyJSONEncoder)  
    
    
    
    
    #histograms for fuel typ
    fig5 = px.histogram(df, x="Fuel_Type")
    graph5JSON=json.dumps(fig5,cls=plotly.utils.PlotlyJSONEncoder)



    #histogram for engine size
    fig6 = px.histogram(df, x="Displacement")
    graph6JSON=json.dumps(fig6,cls=plotly.utils.PlotlyJSONEncoder)



    #3d Scatter
    sns.pairplot(df,vars=[ 'Displacement', 'Mileage', 'Power', 'On_road_price'], hue= 'Fuel_Type',
                palette=sns.color_palette('plasma_r',n_colors=4),diag_kind='kde',height=2, aspect=1.8)

    fig7 = px.scatter_3d(df, x='On_road_price', z='Power', y ='Mileage',color='Manufacturer', width=1000, height=900)
    fig7.update_layout(showlegend=True)
    graph7JSON=json.dumps(fig7,cls=plotly.utils.PlotlyJSONEncoder)


    #scatter for power, engine, bodytype with linear regression
    X = df.Displacement.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, df.Power)
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    fig8 = px.scatter(df, x="Displacement", y="Power", color="Body_Type",
                    hover_data=['Power'])
    fig8.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
    graph8JSON=json.dumps(fig8,cls=plotly.utils.PlotlyJSONEncoder)


    #scatter for power, bodytype, Price with linear regression
    X = df.On_road_price.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, df.Power)
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    fig9 = px.scatter(df, x="On_road_price", y="Power", color="Body_Type",
                    hover_data=['Power'])
    fig9.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
    graph9JSON=json.dumps(fig9,cls=plotly.utils.PlotlyJSONEncoder)

    #Power mileage scatters with linear regression
    X = df.Mileage.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, df.Power)
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    fig10 = px.scatter(df, x="Mileage", y="Power", facet_col="Fuel_Type")
    fig10.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
    graph10JSON=json.dumps(fig10,cls=plotly.utils.PlotlyJSONEncoder)



    X = df.Power.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, df.Mileage)
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    fig11 = px.scatter(df, x ="Power", y ="Mileage" , color ="Manufacturer")
    fig11.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
    graph11JSON=json.dumps(fig11,cls=plotly.utils.PlotlyJSONEncoder)


    #pairplot of mileage,power, on road price and fuel type
    fig12 = px.scatter_matrix(df,
        dimensions=["Mileage", "Displacement", "Power", "On_road_price"],
        color="Fuel_Type")
    fig12.update_layout(showlegend=True)
    graph12JSON=json.dumps(fig12,cls=plotly.utils.PlotlyJSONEncoder)



    #pairplot of length,widht, height and wheelbase
    fig13 = px.scatter_matrix(df,
        dimensions=["Length", "Width", "Height", "Wheelbase"],
        color="Body_Type")
    fig13.update_layout(showlegend=True)
    graph13JSON=json.dumps(fig13,cls=plotly.utils.PlotlyJSONEncoder)


    return render_template("specifications.html",graph1JSON=graph1JSON,graph2JSON=graph2JSON,graph3JSON=graph3JSON,
                           graph4JSON=graph4JSON,graph5JSON=graph5JSON,graph6JSON=graph6JSON,
                           graph7JSON=graph7JSON,graph8JSON=graph8JSON,graph9JSON=graph9JSON,
                           graph10JSON=graph10JSON,graph11JSON=graph11JSON,graph12JSON=graph12JSON,
                           graph13JSON=graph13JSON)


