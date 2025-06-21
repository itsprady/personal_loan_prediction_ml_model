import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.header("Exploratory Data Analysis & Insights")

@st.cache_data
def load_data():
    df_cust = pd.read_csv("customer.csv")
    df_card = pd.read_csv("credit_card.csv")
    df = pd.merge(df_cust, df_card, on="Client_Num", how="inner")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# 1. Correlation heatmap for customer satisfaction
st.subheader("1. Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# 1. Boxplot of satisfaction by education
st.subheader("Satisfaction by Education Level")
fig, ax = plt.subplots()
sns.boxplot(x='Education_Level', y='Cust_Satisfaction_Score', data=df, ax=ax)
st.pyplot(fig)

# ... (add the rest of your Data Insights plots here, as in your previous tab3 code)
