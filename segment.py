from flask import Blueprint, render_template

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression


segment = Blueprint("segment",__name__)

@segment.route("/segment")

def seg():

    #Load customers data
    customersdata = pd.read_csv("segment.csv")


    #creating dataset for the csv file
    df=pd.DataFrame(customersdata)

    #basic cluster 
    fig1 = px.scatter_3d(df, x='Salary', y='Price', z='Age')
    graph1JSON=json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)




    # Define K-means model
    kmeans_model = KMeans(init='k-means++',  max_iter=400, random_state=42)

    # Train the model
    kmeans_model.fit(customersdata[['Salary','Price','Age']])

    # Create the K means model for different values of K
    def try_different_clusters(K, data):
        
        cluster_values = list(range(1, K+1))
        inertias=[]
        
        for c in cluster_values:
            model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
            model.fit(data)
            inertias.append(model.inertia_)
        
        return inertias

    # Find output for k values between 1 to 12 
    outputs = try_different_clusters(12, customersdata[['Salary','Price','Age']])
    distances = pd.DataFrame({"clusters": list(range(1, 13)),"sum of squared distances": outputs})


    # Finding optimal number of clusters k
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

    fig2.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),                  
                    xaxis_title="Number of clusters",
                    yaxis_title="Sum of squared distances",
                    title_text="Finding optimal number of clusters using elbow method")
    graph2JSON=json.dumps(fig2,cls=plotly.utils.PlotlyJSONEncoder)


    # Re-Train K means model with k=5
    kmeans_model_new = KMeans(n_clusters = 5,init='k-means++',max_iter=400,random_state=42)

    kmeans_model_new.fit_predict(customersdata[['Salary','Price','Age']])

    # Create data arrays
    cluster_centers = kmeans_model_new.cluster_centers_
    data = np.expm1(cluster_centers)
    points = np.append(data, cluster_centers, axis=1)
    points

    # Add "clusters" to customers data
    points = np.append(points, [[0], [1], [2], [3], [4]], axis=1)
    customersdata["clusters"] = kmeans_model_new.labels_


    #visualize clusters
    fig3 = px.scatter_3d(customersdata,
                        color='clusters',
                        x="Salary",
                        y="Price",
                        z="Age",            
                        category_orders = {"clusters": ["0", "1", "2", "3", "4"]}                    
                        )
    fig3.update_layout()
    graph3JSON=json.dumps(fig3,cls = plotly.utils.PlotlyJSONEncoder)
    

    #Linear Regression for salary and income
    X = df.Salary.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, df.Price)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig4 = px.scatter(df, x='Salary', y='Price', opacity=0.65)
    fig4.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
    graph4JSON=json.dumps(fig4,cls = plotly.utils.PlotlyJSONEncoder)

    #histogram for loan representation
    
    
    fig5 = px.histogram(df, x="Loan", color="Price")
    graph5JSON=json.dumps(fig5,cls = plotly.utils.PlotlyJSONEncoder)


    fig6 = px.histogram(df, x="Personal loan")
    graph6JSON=json.dumps(fig6,cls = plotly.utils.PlotlyJSONEncoder)
    
    fig7 = px.histogram(df, x="House Loan")
    graph7JSON=json.dumps(fig7,cls = plotly.utils.PlotlyJSONEncoder)


    ## Scatterplot with loans 

    fig8 = px.scatter(df, x="Salary", y="Price", color="House Loan",
                    hover_data=['Salary'])
    graph8JSON=json.dumps(fig8,cls = plotly.utils.PlotlyJSONEncoder)
        
    
    ## Box plot
    fig9 = px.box(df, x="Age", y="No of Dependents", points="all")
    graph9JSON=json.dumps(fig9,cls = plotly.utils.PlotlyJSONEncoder)
    
    ##histogram for profession
    fig10 = px.histogram(df, x="Profession")
    graph10JSON=json.dumps(fig10,cls = plotly.utils.PlotlyJSONEncoder)
    
    
    ## Facetted Scatter
    fig11 = px.scatter(df, x="Salary", y="Price", facet_col="Profession")
    graph11JSON=json.dumps(fig11,cls = plotly.utils.PlotlyJSONEncoder)
    
    
    
    ## 2D cluster for income and time spent on social media
    fig12 = px.scatter(df, x="Salary", y="Time")
    graph12JSON=json.dumps(fig12,cls = plotly.utils.PlotlyJSONEncoder)

    
    # Define K-means model
    kmeans_model = KMeans(init='k-means++',  max_iter=400, random_state=42)

    # Train the model
    kmeans_model.fit(customersdata[['Time','Salary']])

    # Create the K means model for different values of K
    def try_different_clusters(K, data):
        
        cluster_values = list(range(1, K+1))
        inertias=[]
        
        for c in cluster_values:
            model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
            model.fit(data)
            inertias.append(model.inertia_)
        
        return inertias

    # Find output for k values between 1 to 12 
    outputs = try_different_clusters(12,customersdata[['Time','Salary']])
    distances = pd.DataFrame({"clusters": list(range(1, 13)),"sum of squared distances": outputs})

    # Finding optimal number of clusters k
    fig13 = go.Figure()
    fig13.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

    fig13.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),                  
                    xaxis_title="Number of clusters",
                    yaxis_title="Sum of squared distances",
                    title_text="Finding optimal number of clusters using elbow method")

    # Re-Train K means model with k=4
    kmeans_model_new = KMeans(n_clusters = 4,init='k-means++',max_iter=400,random_state=42)

    kmeans_model_new.fit_predict(customersdata[['Time','Salary']])

    # Create data arrays
    cluster_centers = kmeans_model_new.cluster_centers_
    data = np.expm1(cluster_centers)
    points = np.append(data, cluster_centers, axis=1)
    points

    # Add "clusters" to customers data
    points = np.append(points, [[0], [1], [2], [3]], axis=1)
    customersdata["clusters"] = kmeans_model_new.labels_


    #visualize clusters
    fig14 = px.scatter(customersdata,
                        color='clusters',
                        x="Time",
                        y="Salary",
                        category_orders = {"clusters": ["0", "1", "2", "3"]}                    
                        )
    fig14.update_layout()
    graph14JSON=json.dumps(fig14,cls = plotly.utils.PlotlyJSONEncoder)
    
    


    
    
    

    return render_template("segment.html",graph1JSON=graph1JSON,graph2JSON=graph2JSON,graph3JSON=graph3JSON,
                           graph4JSON=graph4JSON,graph5JSON=graph5JSON,graph6JSON=graph6JSON,
                           graph7JSON=graph7JSON,graph8JSON=graph8JSON,graph9JSON=graph9JSON,
                           graph10JSON=graph10JSON,graph11JSON=graph11JSON,graph12JSON=graph12JSON,
                           graph14JSON=graph14JSON)


