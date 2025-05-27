import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import pycountry
#ss
#yes
#Load Dataset
df = pd.read_csv('dataset.csv')

#Columns to group by
grouping_columns = [
    'payment_method', 'currency', 'issuer_country',
    'shopper_country', 'risk_scoring', 'shopper_interaction', 'issuer_name', 'merchant_account', "amount_eur", "liability_shift", "pos_entry_mode", "acquirer", "avs_response", "cvc2_response", "3d_directory_response", "3d_authentication_response", "payment_method_variant", "global_card_brand", "3ds_version"
]

#Convert to datetime format for creation date
if 'creation_date' in df.columns:
    df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
    df = df.dropna(subset=['creation_date'])
    df['date_only'] = df['creation_date'].dt.date
    df['datetime_halfhour'] = df['creation_date'].dt.floor('30min')
    df['datetime_tenmins'] = df['creation_date'].dt.floor('10min')


#Remove NA rows
df = df.dropna(subset=['acquirer_response'])

#Convert issuer_country and shopper_country from alpha-2 to alpha-3 for mapping proper countries
country_map = {country.alpha_2: country.alpha_3 for country in pycountry.countries if hasattr(country, 'alpha_2') and hasattr(country, 'alpha_3')}
df['issuer_country_alpha3'] = df['issuer_country'].map(country_map)
df['shopper_country_alpha3'] = df['shopper_country'].map(country_map)

#Convert to EUR using this dictionary
exchange_rates = {
    'USD': 1.0917, 'JPY': 160.33, 'GBP': 0.85708, 'DKK': 7.4602, 'SEK': 11.4955,
    'MXN': 20.5652, 'HUF': 395.20, 'PLN': 4.3253, 'CZK': 25.234, 'CHF': 0.9435,
    'RON': 4.9769, 'NOK': 11.8295, 'BGN': 1.9558, 'RSD': 117.02, 'ALL': 100.13,
    'MKD': 61.514, 'UAH': 44.950, 'BAM': 1.9574, 'EUR': 1.0
}
df['amount_eur'] = df.apply(lambda row: row['amount'] / exchange_rates.get(row['currency'], np.nan), axis=1)

#Initialize Streamlit app
st.set_page_config(layout="wide")
st.title("Transaction Dashboard")

#Set up filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range:",
    [df['date_only'].min(), df['date_only'].max()]
)

#Filters for Currency and Payment Method
#You can add more filters but I just used these two
currency_filter = st.sidebar.multiselect("Select Currency:", options=df['currency'].dropna().unique(), default=df['currency'].dropna().unique())
payment_method_filter = st.sidebar.multiselect("Select Payment Method:", options=df['payment_method'].dropna().unique(), default=df['payment_method'].dropna().unique())

#Apply filters
if len(date_range) == 2:
    df = df[(df['date_only'] >= date_range[0]) & (df['date_only'] <= date_range[1])]
df = df[df['currency'].isin(currency_filter)]
df = df[df['payment_method'].isin(payment_method_filter)]

#One-variable grouping dashboard
st.header("Approval Ratio and Counts by Single Variable")
single_group_var = st.selectbox("Select one variable to group by:", options=grouping_columns)

if single_group_var == 'amount_eur':
    df['amount_bucket'] = (df['amount_eur'] // 20 * 20).astype(int)
    single_group = df.groupby('amount_bucket').agg(
        total_transactions=('acquirer_response', 'count'),
        approved_transactions=('acquirer_response', lambda x: (x == 'APPROVED').sum())
    ).reset_index()
    single_group['approval_ratio'] = single_group['approved_transactions'] / single_group['total_transactions']

    fig_single_ratio = px.bar(
        single_group,
        x='amount_bucket',
        y='approval_ratio',
        title="Approval Ratio by Amount Bucket (EUR)",
        labels={'approval_ratio': 'Approval Ratio', 'amount_bucket': 'Amount Bucket (EUR)'}
    )
    fig_single_ratio.update_layout(yaxis_tickformat='.0%')

    fig_single_count = px.bar(
        single_group,
        x='amount_bucket',
        y='approved_transactions',
        title="Approved Transactions Count by Amount Bucket (EUR)",
        labels={'approved_transactions': 'Approved Transactions', 'amount_bucket': 'Amount Bucket (EUR)'}
    )
else:
    single_group = df.groupby(single_group_var).agg(
        total_transactions=('acquirer_response', 'count'),
        approved_transactions=('acquirer_response', lambda x: (x == 'APPROVED').sum()),
        total_amount_eur=('amount_eur', 'sum')
    ).reset_index()
    single_group['approval_ratio'] = single_group['approved_transactions'] / single_group['total_transactions']

    fig_single_ratio = px.bar(
        single_group,
        x=single_group_var,
        y='approval_ratio',
        title=f"Approval Ratio by {single_group_var}",
        labels={'approval_ratio': 'Approval Ratio'}
    )
    fig_single_ratio.update_layout(yaxis_tickformat='.0%')

    fig_single_count = px.bar(
        single_group,
        x=single_group_var,
        y='approved_transactions',
        title=f"Approved Transactions Count by {single_group_var}",
        labels={'approved_transactions': 'Approved Transactions'}
    )

st.plotly_chart(fig_single_ratio)
st.plotly_chart(fig_single_count)
st.dataframe(single_group)


#Two-variable grouping dashboard
st.header("Approval Ratio and Counts by Two Variables")
st.markdown("Select exactly two variables to group by and analyze approval ratios.")

two_group_vars = st.multiselect(
    "Select 2 variables for grouping:",
    options=grouping_columns,
    default=['payment_method', 'currency']
)

if len(two_group_vars) != 2:
    st.warning("Please select exactly 2 variables.")
else:
    grouped = df.groupby(two_group_vars).agg(
        total_transactions=('acquirer_response', 'count'),
        approved_transactions=('acquirer_response', lambda x: (x == 'APPROVED').sum())
    ).reset_index()

    grouped['approval_ratio'] = grouped['approved_transactions'] / grouped['total_transactions']

    fig_ratio = px.bar(
        grouped,
        x=two_group_vars[0],
        y='approval_ratio',
        color=two_group_vars[1],
        barmode='group',
        title=f"Approval Ratio by {two_group_vars[0]} and {two_group_vars[1]}",
        labels={'approval_ratio': 'Approval Ratio'}
    )
    fig_ratio.update_layout(yaxis_tickformat='.0%')

    fig_count = px.bar(
        grouped,
        x=two_group_vars[0],
        y='approved_transactions',
        color=two_group_vars[1],
        barmode='group',
        title=f"Approved Transactions Count by {two_group_vars[0]} and {two_group_vars[1]}",
        labels={'approved_transactions': 'Approved Transactions'}
    )

    st.plotly_chart(fig_ratio)
    st.plotly_chart(fig_count)
    st.dataframe(grouped)

#Geographic dashboard: Approval ratio by issuer_country
st.header("Geographic Approval Ratio and Counts by Issuer Country")

df_issuer_geo = df.dropna(subset=['issuer_country_alpha3'])
issuer_geo_group = df_issuer_geo.groupby('issuer_country_alpha3').agg(
    total_transactions=('acquirer_response', 'count'),
    approved_transactions=('acquirer_response', lambda x: (x == 'APPROVED').sum())
).reset_index()
issuer_geo_group['approval_ratio'] = issuer_geo_group['approved_transactions'] / issuer_geo_group['total_transactions']

fig_issuer_map_ratio = px.choropleth(
    issuer_geo_group,
    locations='issuer_country_alpha3',
    locationmode='ISO-3',
    color='approval_ratio',
    color_continuous_scale='Blues',
    title='Approval Ratio by Issuer Country',
    labels={'approval_ratio': 'Approval Ratio'}
)
fig_issuer_map_ratio.update_layout(geo=dict(showframe=False, showcoastlines=False))
fig_issuer_map_ratio.update_coloraxes(colorbar_tickformat='.0%')

fig_issuer_map_count = px.choropleth(
    issuer_geo_group,
    locations='issuer_country_alpha3',
    locationmode='ISO-3',
    color='approved_transactions',
    color_continuous_scale='Purples',
    title='Approved Transactions Count by Issuer Country',
    labels={'approved_transactions': 'Approved Transactions'}
)

st.plotly_chart(fig_issuer_map_ratio)
st.plotly_chart(fig_issuer_map_count)
st.dataframe(issuer_geo_group)

#Geographic dashboard: Approval ratio by shopper_country
st.header("Geographic Approval Ratio and Counts by Shopper Country")

df_shopper_geo = df.dropna(subset=['shopper_country_alpha3'])
shopper_geo_group = df_shopper_geo.groupby('shopper_country_alpha3').agg(
    total_transactions=('acquirer_response', 'count'),
    approved_transactions=('acquirer_response', lambda x: (x == 'APPROVED').sum())
).reset_index()
shopper_geo_group['approval_ratio'] = shopper_geo_group['approved_transactions'] / shopper_geo_group['total_transactions']

fig_shopper_map_ratio = px.choropleth(
    shopper_geo_group,
    locations='shopper_country_alpha3',
    locationmode='ISO-3',
    color='approval_ratio',
    color_continuous_scale='Greens',
    title='Approval Ratio by Shopper Country',
    labels={'approval_ratio': 'Approval Ratio'}
)
fig_shopper_map_ratio.update_layout(geo=dict(showframe=False, showcoastlines=False))
fig_shopper_map_ratio.update_coloraxes(colorbar_tickformat='.0%')

fig_shopper_map_count = px.choropleth(
    shopper_geo_group,
    locations='shopper_country_alpha3',
    locationmode='ISO-3',
    color='approved_transactions',
    color_continuous_scale='Oranges',
    title='Approved Transactions Count by Shopper Country',
    labels={'approved_transactions': 'Approved Transactions'}
)

st.plotly_chart(fig_shopper_map_ratio)
st.plotly_chart(fig_shopper_map_count)
st.dataframe(shopper_geo_group)


#Time series analysis (aggregated to 10-minute intervals)
st.header("Time Series Analysis of Approval Ratios and Counts (10-Minute Intervals)")
time_series = df.groupby('datetime_tenmins').agg(
    total_transactions=('acquirer_response', 'count'),
    approved_transactions=('acquirer_response', lambda x: (x == 'APPROVED').sum())
).reset_index()
time_series['approval_ratio'] = time_series['approved_transactions'] / time_series['total_transactions']

fig_ts_ratio = px.line(
    time_series,
    x='datetime_tenmins',
    y='approval_ratio',
    title="Approval Ratio Over Time (Per 10 Minutes)",
    labels={'approval_ratio': 'Approval Ratio', 'datetime_tenmins': 'Datetime'}
)
fig_ts_ratio.update_layout(yaxis_tickformat='.0%')

fig_ts_count = px.line(
    time_series,
    x='datetime_tenmins',
    y='approved_transactions',
    title="Approved Transactions Count Over Time (Per 10 Minutes)",
    labels={'approved_transactions': 'Approved Transactions', 'datetime_tenmins': 'Datetime'}
)

st.plotly_chart(fig_ts_ratio)
st.plotly_chart(fig_ts_count)
st.dataframe(time_series)

