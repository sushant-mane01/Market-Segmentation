import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
# Set the background color to blue
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Market Segmentation")

import os

# Change the working directory
os.chdir("D:/clusteringProject")


with st.form("my_form"):
    # Create a dropdown to select the model type
    model_type_options = ["K-means", "Agglomerative", "DBSCAN"]
    model_type = st.selectbox("Select Model Type", model_type_options)
    
    # Input fields for user data
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    
    cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
    
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Trx', step=1)
    purchases_trx = st.number_input(label='Purchases TRX', step=1)
    credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")

    prc_full_payment = st.number_input(label='PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)

    data = [[balance, balance_frequency, purchases,  cash_advance,
             purchases_frequency, cash_advance_frequency,
             cash_advance_trx, purchases_trx, credit_limit, payments, prc_full_payment, tenure]]

    submitted = st.form_submit_button("Submit")

if submitted:
    if model_type == "K-means":
        loaded_model = pickle.load(open("D:/clusteringProject/final_model.sav", 'rb'))
        df = pd.read_csv("Clustered_Customer_Data.csv")
    elif model_type == "Agglomerative":
        loaded_model = pickle.load(open("D:/clusteringProject/agg_final_model.sav", 'rb'))
        df = pd.read_csv("agg_Clustered_Customer_Data.csv")
    elif model_type == "DBSCAN":
        loaded_model = pickle.load(open("D:/clusteringProject/dbscan_final_model.sav", 'rb'))
        df = pd.read_csv("dbscan_Clustered_Customer_Data.csv")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    cluster = loaded_model.predict(data)[0]
    st.write('Data Belongs to Cluster', cluster)
    
    if 'Cluster' in df.columns:
        cluster_df = df[df['Cluster'] == cluster]
    
        for c in cluster_df.drop(['Cluster'], axis=1):
            if "Unnamed" not in c:

                st.subheader(f"Histogram for {c}")
                sns.histplot(cluster_df[c], kde=True)
                st.pyplot()
    else:
        st.write('Cluster column not found in the data.')