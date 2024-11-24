import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import csv
import openai
import os
import io
from io import StringIO
import gspread
import requests
from google.cloud import storage
import json
import fitz
import pymongo
from pymongo import MongoClient
import boto3
from botocore.exceptions import NoCredentialsError
#from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage,HumanMessage,SystemMessage
from langchain.agents import load_tools, initialize_agent, AgentType
#from langchain.chat_models import ChatOpenAI
#from langchain.llms import openai
import tempfile
from PyPDF2 import PdfFileReader
from dotenv import load_dotenv
load_dotenv()

# Setup OpenAI API Key
#with open('apikey.txt','r') as file:
#    openai.api_key = file.read()
openai.api_key = os.environ.get('openai_key')



bucket_name = r'streamlit-fintech-app'
s3 = boto3.client('s3', aws_access_key_id=os.environ.get('aws_access_key_id'), aws_secret_access_key=os.environ.get('aws_secret_access_key'))

# Function to list files in S3 bucket
def list_s3_files(bucket):
    try:
        response = s3.list_objects_v2(Bucket=bucket)
        if 'Contents' in response:
            return [content['Key'] for content in response.get('Contents', [])]#response['Contents']]
        else:
            return []
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return []

files = list_s3_files(bucket_name)


temp_dir = tempfile.mkdtemp()
            
def download_s3_file(bucket, file_key, download_path):
    try:
        s3.download_file(bucket, file_key, download_path)
        return True
    except Exception as e:
        st.error(f"Error downloading {file_key}: {e}")
        return False
    

# Function to read PDF file and analyze content
def read_pdf(file_path):
    content = ""
    with open(file_path, 'rb') as f:
        reader = PdfFileReader(f)
        for page_num in range(reader.getNumPages()):
            page = reader.getPage(page_num)
            content += page.extract_text()
    return content


def read_csv_from_s3(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(data))
        return df
    except Exception as e:
        print(f"Error reading {key}: {e}")
        return None

csv_files = [file for file in files if file.endswith('.csv')]

dataframes = []
for csv_file in csv_files:
    df = read_csv_from_s3(bucket_name, csv_file)
    if df is not None:
        dataframes.append(df)

columns =[
    'EMAIL',
    'Industry'
    'Company Name',
    'Financial Leverage',
    'Interest Coverage Ratio',
    'Current Ratio',
    'Quick Ratio',
    'Cash Ratio',
    'Operating Cash Flow Ratio',
    'Operating Cash Flow to Total Debt',
    'Assets Turnover',
    'ROA',
    'ROE',
    'Gross Profit Margin',
    'Net Profit Margin',
    'Loan-to-Value (LTV) Ratio',
    'Fixed Charge Coverage Ratio'
]

df_r= pd.DataFrame(columns=columns)


for i, df in enumerate(dataframes):
    print(f"DataFrame {i+1}:")
    print(df.head())
    loan_size = float()
    asset_value = float()
    total_liabilities= df['TOTAL CURRENT LIABILITIES'] + df['TOTAL LONG-TERM LIABILITIES']
    #total_a= df['TOTAL CURRENT ASSETS'] + df['TOTAL LONG-TERM ASSETS']
    quick = df['TOTAL CURRENT ASSETS']- df['INVENTORY']
    cash = df['CASH & BANK']
    ebit = df['QUARTER EBITDA'] - df['QUARTER DEPRECIATION']
    fixed_charges = df['QUARTER FINANCIAL EXPENSES'] + df['CAPEX']
    fixed_charge_coverage_ratio = (ebit + fixed_charges) / fixed_charges
    ltv = loan_size / asset_value
    operating_cash_flow = df['QUARTER EBITDA'] + df['QUARTER DEPRECIATION'] - df['CAPEX']


    df_r['EMAIL'] = df['EMAIL']
    df_r['Company Name'] = df['COMPANY NAME']
    df_r['Industry'] = df['INDUSTRY']
    df_r['Financial Leverage'] = df['TOTAL ASSETS'] / df['TOTAL EQUITY'] 
    df_r['Interest Coverage Ratio'] = ebit / ['QUARTER FINANCIAL EXPENSES']
    df_r['Current Ratio'] = df['TOTAL CURRENT ASSETS'] / df['TOTAL CURRENT LIABILITIES']
    df_r['Quick Ratio'] = quick / df['TOTAL CURRENT LIABILITIES']
    df_r['Cash Ratio'] = cash / df['TOTAL CURRENT LIABILITIES']
    df_r['Fixed Charge Coverage Ratio'] = fixed_charge_coverage_ratio
    df_r['Gross Profit Margin'] = df['Gross Profit Margin']
    df_r['Net Profit Margin'] = df['QUARTER NET INCOME'] / df['QUARTER SALES']
    df_r['Loan-to-Value (LTV) Ratio'] = ltv
    df_r['Operating Cash Flow Ratio'] = operating_cash_flow / df['TOTAL CURRENT LIABILITIES']
    df_r['Operating Cash Flow to Total Debt Ratio'] = operating_cash_flow / total_liabilities
    df_r['Assets Turnover'] = df['QUARTER SALES'] / df['TOTAL ASSETS']
    df_r['ROA'] = df['QUARTER NET INCOME'] / df['TOTAL ASSETS']
    df_r['ROE'] = df_r['Financial Leverage'] * df_r['Assets Turnover'] * df_r['Net Profit Margin']

    print(df_r)



sheet_id='1pe6yUobWyFMD_yUx2TU5YBw-g8fBXgsvWXTrOKPznLw'  #"10Urm62IQmuVo3NeCPDDfjq7pjI0Cb3rosCS43_2GQEY"
sheet_name= ['LEAD','USERS', 'RATIOS','COMPANY DATA', 'FINANCIAL DATA']#"lads"
url_csv="https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}"
# Function to fetch Google Sheet data and convert it to a DataFrame

def fetch_google_sheet():#(sheet_url):
    csv_url = url_csv.format(sheet_id,sheet_name[4]) #sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    response = requests.get(csv_url)
    response.raise_for_status()  # Ensure the request was successful
    df = pd.read_csv(io.StringIO(response.text))
    #df = pd.read_excel('financial_gpt.xlsx',sheet_name=sheet_name[4])
    return df

# Function to convert DataFrame to CSV and provide download link
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

df2 = fetch_google_sheet()


bucket_name = 'your_bucket_name'  # Replace with your bucket name
files = list_s3_files(bucket_name)
temp_dir = tempfile.mkdtemp()

for file_key in files:
    if file_key.endswith('.pdf'):
        download_path = os.path.join(temp_dir, os.path.basename(file_key))
        if download_s3_file(bucket_name, file_key, download_path):
            pdf_content = read_pdf(download_path)
            # Analyze PDF content
            print(f"Content of {file_key}:\n{pdf_content}")


# Cleanup
#import shutil
#shutil.rmtree(temp_dir)