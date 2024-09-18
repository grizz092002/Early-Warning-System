import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

def process_data(municipality=None):
    try:
        data = pd.read_csv('C:/Users/Joseph Collantes/OneDrive/Desktop/THESIS/Dengue-Cases.csv', encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv('C:/Users/Joseph Collantes/OneDrive/Desktop/THESIS/Dengue-Cases.csv', encoding='ISO-8859-1')

    municipality_column = 'Muncity'
    data['Age'] = data['AgeYears'] + data['AgeMons'] / 12 + data['AgeDays'] / 365
    data['DAdmit'] = pd.to_datetime(data['DAdmit'], errors='coerce', dayfirst=True)
    data.dropna(subset=['DAdmit'], inplace=True)
    data['AdmitMonth'] = data['DAdmit'].dt.month
    data['AdmitYear'] = data['DAdmit'].dt.year

    bins = [0, 1, 4, 12, 19, 39, 59, 100]
    labels = ['Infant (0-1 yr)', 'Toddler (2-4 yrs)', 'Child (5-12 yrs)', 'Teen (13-19 yrs)', 'Adult (20-39 yrs)', 'Middle Age Adult (40-59 yrs)', 'Senior Adult (60+)']
    data['AdmitMonthYear'] = data['DAdmit'].dt.strftime('%Y-%m')
    data['AgeCategory'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
    
    age_gender_counts = data.groupby(['AgeCategory', 'Sex'], observed=False).size().reset_index(name='Count')

    if municipality:
        data = data[data['Muncity'] == municipality]
    
    barangay_cases = data.groupby(['Muncity', 'Barangay']).size().reset_index(name='Cases')
    plots = {}
    unique_municipalities = data['Muncity'].unique()

    font_style = {'family': 'Arial', 'size': 13, 'color': 'black'}

    for muncity in unique_municipalities:
        muncity_data = barangay_cases[barangay_cases['Muncity'] == muncity]
        fig = px.bar(muncity_data, y='Barangay', x='Cases', title=f'Cases per Barangay in {muncity}', color_discrete_sequence=['#1C5730'])
        fig.update_layout(width=1200, height=750, dragmode=False, font=font_style)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        plots[muncity] = fig.to_html(full_html=False)
        
    month_counts = data['AdmitMonth'].value_counts().sort_index()
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_counts.index = month_counts.index.map(lambda x: month_names[x - 1] if x in range(1, 13) else '')

    month_distribution_fig = px.bar(x=month_counts.index, y=month_counts.values, color_discrete_sequence=['#1C5730'])
    month_distribution_fig.update_layout(dragmode=False, font=font_style)
    month_distribution_fig.update_xaxes(title='Month', fixedrange=True)
    month_distribution_fig.update_yaxes(title='Cases', fixedrange=True)
    month_distribution_html = month_distribution_fig.to_html(full_html=False)

    age_gender_fig = px.bar(age_gender_counts, x='AgeCategory', y='Count', color='Sex', barmode='group', labels={'Count': 'Cases', 'AgeCategory': 'Age Category'}, color_discrete_map={'F': '#81A263', 'M': '#1C5730'})
    age_gender_fig.update_layout(bargap=0.2, dragmode=False, font=font_style)
    age_gender_fig.update_xaxes(fixedrange=True)
    age_gender_fig.update_yaxes(fixedrange=True)
    age_gender_html = age_gender_fig.to_html(full_html=False)
    
    cases_per_municipality_data = data['Muncity'].value_counts().reset_index()
    cases_per_municipality_data.columns = ['Muncity', 'Cases']
    
    cases_per_municipality_data = cases_per_municipality_data.sort_values(by='Muncity')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=cases_per_municipality_data['Cases'],
        x=cases_per_municipality_data['Muncity'],
        marker_color='#1C5730',
        name='Cases'
    ))

    fig.update_layout(
        title='Total Distribution of Dengue Cases by Municipality (2019-2023)',  # Add title here
        showlegend=False,
        dragmode=False,
        font=font_style,
        xaxis_title='Municipality',
        yaxis_title='Cases',
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True)
    )

    cases_per_municipality_html = fig.to_html(full_html=False)
    cases_per_municipality = cases_per_municipality_data.to_dict(orient='records')

    monthly_admission_data = data.groupby(['Muncity', 'AdmitYear', 'AdmitMonth']).size().reset_index(name='Cases')
    
    months_full = pd.DataFrame({'AdmitMonth': range(1, 13), 'MonthName': month_names})
    monthly_admission_data = monthly_admission_data.merge(months_full, how='right', on='AdmitMonth').fillna({'Cases': 0})

    monthly_admission_pivot = monthly_admission_data.pivot_table(index=['Muncity', 'AdmitYear'], columns='MonthName', values='Cases', fill_value=0)

    # Plot for monthly distribution of dengue cases per municipality
    monthly_admission_plots = {}
    for year in data['AdmitYear'].unique():
        yearly_data = monthly_admission_data[monthly_admission_data['AdmitYear'] == year]
        fig = go.Figure()
        for muncity in unique_municipalities:
            muncity_data = yearly_data[yearly_data['Muncity'] == muncity]
            fig.add_trace(go.Scatter(
                x=muncity_data['MonthName'],
                y=muncity_data['Cases'],
                mode='lines+markers',
                name=muncity
            ))
        fig.update_layout(
            title=f'Monthly Dengue Cases Distribution for Each Municipality in {year}',
            xaxis_title='Month',
            yaxis_title='Cases',
            xaxis=dict(type='category'),
            yaxis=dict(title='Number of Cases')
        )
        monthly_admission_plots[year] = fig.to_html(full_html=False)

    return (month_distribution_html, 
            age_gender_html, 
            cases_per_municipality_html, 
            cases_per_municipality, 
            plots,
            monthly_admission_plots)
