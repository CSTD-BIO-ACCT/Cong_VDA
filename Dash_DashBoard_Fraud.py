import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import numpy as np
import pycountry

#Load dataset
df = pd.read_csv('dataset.csv')

#Define relevant columns
grouping_columns = [
    'payment_method', 'currency', 'issuer_country',
    'shopper_country', 'risk_scoring', 'shopper_interaction', 'issuer_name', 'merchant_account', "amount_eur", "liability_shift", "pos_entry_mode", "acquirer", "avs_response", "cvc2_response", "3d_directory_response", "3d_authentication_response", "payment_method_variant", "global_card_brand", "3ds_version"
]

#Convert creation_date to datetime if available
if 'creation_date' in df.columns:
    df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
    df = df.dropna(subset=['creation_date'])
    df['date_only'] = df['creation_date'].dt.date
    df['datetime_minute'] = df['creation_date'].dt.floor('T')
    df['datetime_10min'] = df['creation_date'].dt.floor('10min')

#Remove rows with missing acquirer_response
df = df.dropna(subset=['acquirer_response'])

#ISO conversion from alpha-2 to alpha-3 for mapping
country_map = {country.alpha_2: country.alpha_3 for country in pycountry.countries if hasattr(country, 'alpha_2') and hasattr(country, 'alpha_3')}
df['issuer_country_alpha3'] = df['issuer_country'].map(country_map)
df['shopper_country_alpha3'] = df['shopper_country'].map(country_map)

#Exchange Rates
eur_conversion_rates = {
    'USD': 1.0917, 'JPY': 160.33, 'GBP': 0.85708, 'DKK': 7.4602, 'SEK': 11.4955,
    'MXN': 20.5652, 'HUF': 395.20, 'PLN': 4.3253, 'CZK': 25.234, 'CHF': 0.9435,
    'RON': 4.9769, 'NOK': 11.8295, 'BGN': 1.9558, 'RSD': 117.02, 'ALL': 100.13,
    'MKD': 61.514, 'UAH': 44.950, 'BAM': 1.9574, 'EUR': 1.0
}
df['amount_eur'] = df.apply(lambda row: row['amount'] / eur_conversion_rates.get(row['currency'], np.nan), axis=1)

#Initialize Dash App
app = dash.Dash(__name__)
app.title = "Fraud Detection Dashboard"

app.layout = html.Div([
    html.H1("Fraud Transaction Analysis Dashboard"),

    html.Label("Select Variable for Single Variable Grouping:"),
    dcc.Dropdown(
        id='single-group-dropdown',
        options=[{'label': col, 'value': col} for col in grouping_columns],
        value='payment_method',
        clearable=False
    ),
    dcc.Graph(id='fraud-ratio-chart'),
    dcc.Graph(id='fraud-count-chart'),

    html.Hr(),
    html.Label("Select Country Dimension for Geographic Fraud Ratio:"),
    dcc.RadioItems(
        id='geo-country-select',
        options=[
            {'label': 'Issuer Country', 'value': 'issuer_country_alpha3'},
            {'label': 'Shopper Country', 'value': 'shopper_country_alpha3'}
        ],
        value='issuer_country_alpha3',
        inline=True
    ),
    dcc.Graph(id='fraud-geo-map'),

    html.Hr(),
    html.Label("Fraud Time Series (10-minute intervals):"),
    dcc.Graph(id='fraud-time-series')
])

@app.callback(
    [Output('fraud-ratio-chart', 'figure'),
     Output('fraud-count-chart', 'figure')],
    [Input('single-group-dropdown', 'value')]
)
def update_charts(group_var):
    if group_var == 'amount_eur':
        df['amount_bucket'] = (df['amount_eur'] // 50 * 50).astype(int)
        grouped = df.groupby('amount_bucket').agg(
            total_transactions=('acquirer_response', 'count'),
            fraud_transactions=('acquirer_response', lambda x: (x == 'FRAUD').sum())
        ).reset_index()
        grouped['fraud_ratio'] = grouped['fraud_transactions'] / grouped['total_transactions']
        x_col = 'amount_bucket'
    else:
        grouped = df.groupby(group_var).agg(
            total_transactions=('acquirer_response', 'count'),
            fraud_transactions=('acquirer_response', lambda x: (x == 'FRAUD').sum())
        ).reset_index()
        grouped['fraud_ratio'] = grouped['fraud_transactions'] / grouped['total_transactions']
        x_col = group_var

        if group_var == 'issuer_name':
            grouped = grouped[grouped['fraud_ratio'] > 0].sort_values(by='fraud_ratio', ascending=True)

    fig_ratio = px.bar(grouped, x=x_col, y='fraud_ratio',
                       title=f"Fraud Ratio by {group_var if group_var != 'amount_eur' else 'Amount Bucket (EUR)'}",
                       labels={x_col: x_col, 'fraud_ratio': 'Fraud Ratio'})
    fig_ratio.update_layout(yaxis_tickformat='.0%')

    fig_count = px.bar(grouped, x=x_col, y='fraud_transactions',
                       title=f"Fraud Count by {group_var if group_var != 'amount_eur' else 'Amount Bucket (EUR)'}",
                       labels={x_col: x_col, 'fraud_transactions': 'Fraud Transactions'})

    return fig_ratio, fig_count

@app.callback(
    Output('fraud-geo-map', 'figure'),
    Input('geo-country-select', 'value')
)
def update_geo_map(country_col):
    geo_group = df.dropna(subset=[country_col])
    grouped = geo_group.groupby(country_col).agg(
        total_transactions=('acquirer_response', 'count'),
        fraud_transactions=('acquirer_response', lambda x: (x == 'FRAUD').sum())
    ).reset_index()
    grouped['fraud_ratio'] = grouped['fraud_transactions'] / grouped['total_transactions']

    fig_map = px.choropleth(
        grouped,
        locations=country_col,
        locationmode='ISO-3',
        color='fraud_ratio',
        color_continuous_scale='Reds',
        title=f"Fraud Ratio by {country_col.replace('_alpha3', '').replace('_', ' ').title()}",
        labels={'fraud_ratio': 'Fraud Ratio'}
    )
    fig_map.update_layout(geo=dict(showframe=False, showcoastlines=False))
    fig_map.update_coloraxes(colorbar_tickformat='.0%')

    return fig_map


@app.callback(
    Output('fraud-time-series', 'figure'),
    Input('single-group-dropdown', 'value')  # Just to trigger update
)
def update_time_series(_):
    time_series = df.groupby('datetime_10min').agg(
        total_transactions=('acquirer_response', 'count'),
        fraud_transactions=('acquirer_response', lambda x: (x == 'FRAUD').sum())
    ).reset_index()
    time_series['fraud_ratio'] = time_series['fraud_transactions'] / time_series['total_transactions']

    fig_ts = px.line(
        time_series,
        x='datetime_10min',
        y='fraud_ratio',
        title="Fraud Ratio Over Time (10-Minute Intervals)",
        labels={'datetime_10min': 'Datetime', 'fraud_ratio': 'Fraud Ratio'}
    )
    fig_ts.update_layout(yaxis_tickformat='.0%')

    return fig_ts

if __name__ == '__main__':
    app.run(debug=True)
