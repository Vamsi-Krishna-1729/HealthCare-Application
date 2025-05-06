import pandas as pd
import plotly.express as px
import dash
from prophet import Prophet
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Initialize Dash app with external stylesheets and suppress callback exceptions
external_stylesheets = [
    "/assets/styles.css",
    "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css",
    "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap",
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Load datasets (update file paths as needed)
patients_df = pd.read_csv('~/HealthCare-Application/Data/patients.csv',
                          parse_dates=['BIRTHDATE', 'DEATHDATE'], dtype=str)
encounters_df = pd.read_csv('~/HealthCare-Application/Data/encounters.csv',
                           parse_dates=['START', 'STOP'], dtype=str)
medications_df = pd.read_csv('~/HealthCare-Application/Data/medications.csv',
                             parse_dates=['START'])
observations_df = pd.read_csv('~/HealthCare-Application/Data/observations.csv',
                              parse_dates=['DATE'])
procedures_df = pd.read_csv('~/HealthCare-Application/Data/procedures.csv')




# Remove timezones from datetime columns
for df in [encounters_df, medications_df, observations_df]:
    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_localize(None)




# Data processing for Costs and Utilization Dashboard
merged_costs_df = medications_df.merge(encounters_df[['Id', 'START', 'ENCOUNTERCLASS']],
                                       left_on='ENCOUNTER', right_on='Id', how='left')
merged_costs_df = merged_costs_df.rename(columns={'START_x': 'MEDICATION_START', 'START_y': 'ENCOUNTER_START'})
encounter_costs = merged_costs_df.groupby('ENCOUNTER').agg({
    'TOTALCOST': 'sum',
    'ENCOUNTER_START': 'first',
    'ENCOUNTERCLASS': 'first'
}).reset_index()
overall_avg_cost = encounter_costs.groupby('ENCOUNTERCLASS')['TOTALCOST'].mean()

merged_costs_df['week'] = merged_costs_df['MEDICATION_START'].dt.to_period('W')
merged_costs_df['month'] = merged_costs_df['MEDICATION_START'].dt.to_period('M')
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_utilization = merged_costs_df.groupby([merged_costs_df['MEDICATION_START'].dt.day_name(), 'ENCOUNTERCLASS']).size() \
    .unstack(fill_value=0).reindex(days_order)
weekly_utilization = merged_costs_df.groupby(['week', 'ENCOUNTERCLASS']).size().unstack(fill_value=0).sort_index()
monthly_utilization = merged_costs_df.groupby(['month', 'ENCOUNTERCLASS']).size().unstack(fill_value=0).sort_index()
color_map = {'emergency': 'red', 'surgery': 'darkblue', 'cardiology': 'orange', 'general': 'purple'}

# Data processing for Patient Analytics Dashboard
merged_analytics_df = encounters_df.merge(patients_df, left_on='PATIENT', right_on='Id', how='left')

def determine_status(row):
    if pd.isna(row['STOP']):
        return 'In-treatment'
    elif not pd.isna(row['DEATHDATE']):
        return 'Expired'
    else:
        return 'Discharged'

merged_analytics_df['STATUS'] = merged_analytics_df.apply(determine_status, axis=1)
merged_analytics_df['DATE'] = pd.to_datetime(merged_analytics_df['START']).dt.date

# Data processing for Patient Information Dashboard
patient_list = patients_df[['Id', 'FIRST', 'LAST']]
options = [{'label': f"{row['FIRST']} {row['LAST']}", 'value': row['Id']} for index, row in patient_list.iterrows()]

# Forecasting function
def forecast_encounters(utilization_df, period='W'):
    df = utilization_df.sum(axis=1).reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = df['ds'].dt.to_timestamp()

    model = Prophet()
    model.fit(df)

    future_periods = 6  # Forecast 6 periods into the future
    if period == 'W':
        future = model.make_future_dataframe(periods=future_periods, freq='W')
    else:
        future = model.make_future_dataframe(periods=future_periods, freq='M')

    forecast = model.predict(future)
    return forecast

overview_layout = html.Div([
    html.H1("Hospital Overview Dashboard", className="display-4 text-center mb-4 text-Primary fw-bold"),
    html.Div([  # Filters row
        html.Div([
            html.Label("📅 Date Range", className="form-label fw-bold"),
            dcc.Dropdown(
                id='overview-date-filter',
                options=[
                    {'label': 'Last 7 days', 'value': '7D'},
                    {'label': 'Last 30 days', 'value': '30D'},
                    {'label': 'All Time', 'value': 'ALL'}
                ],
                value='ALL', clearable=False, className="form-select shadow-sm"
            )
        ], className="col-md-3 mb-3"),

        html.Div([  # Encounter Class Filter
            html.Label("🏥 Encounter Class", className="form-label fw-bold"),
            dcc.Dropdown(
                id='overview-encounter-class-filter',
                options=[{'label': 'All', 'value': 'ALL'}] + [
                    {'label': enc_class, 'value': enc_class} for enc_class in encounters_df['ENCOUNTERCLASS'].unique()
                ],
                value='ALL', clearable=False, className="form-select shadow-sm"
            )
        ], className="col-md-3 mb-3"),
        html.Div([  # Gender Filter
            html.Label("⚧ Gender", className="form-label fw-bold"),
            dcc.Dropdown(
                id='overview-gender-filter',
                options=[
                    {'label': 'All', 'value': 'ALL'},
                    {'label': 'Male', 'value': 'M'},
                    {'label': 'Female', 'value': 'F'}
                ],
                value='ALL', clearable=False, className="form-select shadow-sm"
            )
        ], className="col-md-3 mb-3"),
    ], className="row bg-light p-3 rounded shadow-sm"),

    html.Div(id='overview-metrics',  # Metrics Section
             className="card shadow-lg p-4 mb-4 bg-light rounded metrics-container d-flex justify-content-around align-items-center"),

    html.Div([  # Remaining Graphs
        html.Div([  # Resource Usage Graph
            html.Div("🛠 Resource Usage", className="card-header bg-success text-white fw-bold"),
            dcc.Graph(id='overview-resource-usage')
        ], className="card shadow-lg p-3 mb-4 col-md-12 graph-container"),
        html.Div([  # Cost Analysis Graph
            html.Div("💰 Cost Per Procedure", className="card-header bg-warning text-dark fw-bold"),
            dcc.Graph(id='overview-cost-analysis')
        ], className="card shadow-lg p-3 col-md-12 graph-container")
    ], className="row")
], className="p-4 bg-light")

# Costs and Utilization Dashboard Layout
costs_layout = html.Div([
    html.H1("Encounters Dashboard", className="display-4 text-center mb-4 text-black fw-bold"),
    html.Div([
        html.Div([
            html.H4("Overall Average Cost per Encounter", className="card-title fw-bold"),
            html.Div([
                html.P([
                    html.Span(f"{enc_class.capitalize()}: ",
                              style={'color': color_map.get(enc_class, 'black'), 'fontWeight': 'bold'}),
                    f"${cost:.2f}"
                ], className="mb-2")
                for enc_class, cost in overall_avg_cost.items()
            ], className="card-body")
        ], className="card shadow-lg p-4 mb-4 bg-light rounded col-md-6"),
        html.Div([
            html.Label("Encounters", className="form-label fw-bold"),
            dcc.Dropdown(
                id='time-aggregation',
                options=[
                    {'label': 'Day of Week', 'value': 'Day of Week'},
                    {'label': 'Week', 'value': 'Week'},
                    {'label': 'Month', 'value': 'Month'}
                ],
                value='Day of Week', clearable=False, className="form-select shadow-sm mb-3"
            ),
            dcc.Graph(id='utilization-graph')
        ], className="card shadow-lg p-4 mb-4 bg-light rounded col-md-6 graph-container")
    ], className="row justify-content-center"),
html.Div([
        html.Label("📈 Forecast Type", className="form-label fw-bold"),
        dcc.Dropdown(
            id='forecast-period',
            options=[
                {'label': 'Weekly', 'value': 'W'},
                {'label': 'Monthly', 'value': 'M'}
            ],
            value='W',
            clearable=False,
            className="form-select shadow-sm mb-3"
        ),
        dcc.Graph(id='forecast-graph')
    ], className="card shadow-lg p-4 mb-4 bg-light rounded")
], className="p-4 bg-light")

analytics_layout = html.Div([
    html.H1("📊 Patient Analytics Dashboard", className="display-4 text-center mb-4 text-black fw-bold"),

    # New Card for Total Patients
    html.Div([
        html.Div([
            html.Div("📊 Total Patients", className="card-header bg-primary text-white fw-bold"),
            html.Div(id='total-patients', className="card-body"),
        ], className="card shadow-lg p-4 mb-4 bg-light rounded col-md-12")
    ], className="row"),

    html.Div([  # Existing filters
        html.Div([  # Search by Name
            html.Label("🔎 Search Patient by Name", className="form-label fw-bold"),
            dcc.Input(id="search-name", type="text", placeholder="Enter Name", className="form-control shadow-sm mb-3")
        ], className="col-md-4"),
        html.Div([  # Department Filter
            html.Label("🏥 Department", className="form-label fw-bold"),
            dcc.Dropdown(
                id="department-filter",
                options=[{"label": dept, "value": dept} for dept in merged_analytics_df["ENCOUNTERCLASS"].unique()],
                placeholder="Select Department", multi=True, className="form-select shadow-sm mb-3"
            )
        ], className="col-md-4"),
        html.Div([  # Status Filter
            html.Label("📌 Status", className="form-label fw-bold"),
            dcc.Dropdown(
                id="status-filter",
                options=[{"label": s, "value": s} for s in merged_analytics_df["STATUS"].unique()],
                placeholder="Select Status", multi=True, className="form-select shadow-sm mb-3"
            )
        ], className="col-md-4"),
        html.Div([  # Gender Filter
            html.Label("⚧ Gender", className="form-label fw-bold"),
            dcc.Dropdown(
                id="gender-filter",
                options=[{"label": gender, "value": gender} for gender in merged_analytics_df["GENDER"].unique()],
                placeholder="Select Gender", multi=True, className="form-select shadow-sm mb-3"
            )
        ], className="col-md-4"),
        html.Div([  # Joined in last X days
            html.Label("📅 Joined in last X days", className="form-label fw-bold"),
            dcc.Input(id="days-filter", type="number", placeholder="Enter Days", min=0, step=1, className="form-control shadow-sm mb-3")
        ], className="col-md-4"),
        html.Div([  # Sort by Date
            html.Label("📜 Sort by Date", className="form-label fw-bold"),
            dcc.Dropdown(
                id="sort-order",
                options=[{"label": "Old to New", "value": "asc"}, {"label": "New to Old", "value": "desc"}],
                placeholder="Select Sorting", className="form-select shadow-sm mb-3"
            )
        ], className="col-md-4")
    ], className="row bg-light p-3 rounded shadow-sm mb-4"),

    # Patient Data Table
    dash_table.DataTable(
        id="patients-table",
        columns=[
            {"name": "Patient", "id": "FIRST"},
            {"name": "Department", "id": "ENCOUNTERCLASS"},
            {"name": "Date", "id": "DATE"},
            {"name": "Status", "id": "STATUS"},
            {"name": "Gender", "id": "GENDER"},
            {"name": "Patient ID", "id": "PATIENT"}
        ],
        page_size=10,
        style_table={"overflowX": "auto", "borderRadius": "12px", "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"},
        style_cell={"textAlign": "left", "padding": "10px", "fontSize": "14px"},
        style_header={"backgroundColor": "#343a40", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
            {"if": {"row_index": "even"}, "backgroundColor": "rgb(255, 255, 255)"}
        ],
        style_as_list_view=True
    )
], className="p-4 bg-light")



# Patient Information Dashboard Layout
info_layout = html.Div([
    html.H1("🩺 Patient Information Dashboard", className="display-4 text-center mb-4 text-black fw-bold"),
    html.Div([
        html.Label("🔎 Select a Patient", className="form-label fw-bold"),
        dcc.Dropdown(
            id="patient-select",
            options=options,
            value=None,
            placeholder="Select a patient",
            className="form-select shadow-sm mb-4"
        )
    ], className="col-md-6 mx-auto"),
    html.Div(id="patient-info-section", children=[], className="mt-4")
], className="p-4 bg-light")

@app.callback(
    [Output('overview-metrics', 'children'),
     Output('overview-resource-usage', 'figure'),
     Output('overview-cost-analysis', 'figure')],
    [Input('overview-date-filter', 'value'),
     Input('overview-encounter-class-filter', 'value'),
     Input('overview-gender-filter', 'value')]
)
def update_overview(date_range, encounter_class, gender):
    filtered_patients = patients_df.copy()
    filtered_encounters = encounters_df.copy()
    filtered_observations = observations_df.copy()
    filtered_procedures = procedures_df.copy()

    # Apply gender filter
    if gender and gender != 'ALL':
        filtered_patients = filtered_patients[filtered_patients['GENDER'] == gender]

    # Filter encounters by patients
    filtered_encounters = filtered_encounters[filtered_encounters['PATIENT'].isin(filtered_patients['Id'])]

    # Apply encounter class filter
    if encounter_class and encounter_class != 'ALL':
        filtered_encounters = filtered_encounters[filtered_encounters['ENCOUNTERCLASS'] == encounter_class]

    # Apply date range filter
    if date_range and date_range != 'ALL':
        cutoff_date = pd.Timestamp.now().tz_localize(None) - pd.to_timedelta(date_range)
        filtered_encounters['START'] = pd.to_datetime(filtered_encounters['START'], errors='coerce')
        filtered_encounters = filtered_encounters.dropna(subset=['START'])
        filtered_encounters['START'] = filtered_encounters['START'].dt.tz_localize(None)
        filtered_encounters = filtered_encounters[filtered_encounters['START'] >= cutoff_date]

    # Filter observations and procedures
    filtered_observations = filtered_observations[filtered_observations['PATIENT'].isin(filtered_encounters['PATIENT'])]
    filtered_procedures = filtered_procedures[filtered_procedures['ENCOUNTER'].isin(filtered_encounters['Id'])]

    # Calculate metrics
    total_patients = max(len(filtered_patients), 1)
    unique_active_patients = filtered_encounters['PATIENT'].nunique()
    bed_occupancy = round((unique_active_patients / total_patients * 100), 2) if total_patients > 0 else 0
    avg_wait_time = round(filtered_encounters['BASE_ENCOUNTER_COST'].astype(float).mean() / 10, 2) if not filtered_encounters.empty else 0
    total_revenue = filtered_encounters['TOTAL_CLAIM_COST'].astype(float).sum() if 'TOTAL_CLAIM_COST' in filtered_encounters.columns else 0

    metrics_display = html.Div([
        html.Div([
            html.I(className="fas fa-user-injured fa-2x text-white"),
            html.H4("Total Patients", className="mt-2 text-white"),
            html.H2(f"{total_patients:,}", className="text-white fw-bold")
        ], className="card bg-primary text-white shadow-lg rounded p-4 col-md-3 mb-4"),
        html.Div([
            html.I(className="fas fa-bed fa-2x text-white"),
            html.H4("Bed Occupancy", className="mt-2 text-white"),
            html.H2(f"{bed_occupancy}%", className="text-white fw-bold")
        ], className="card bg-success text-white shadow-lg rounded p-4 col-md-3 mb-4"),
        html.Div([
            html.I(className="fas fa-stopwatch fa-2x text-white"),
            html.H4("Avg Wait Time", className="mt-2 text-white"),
            html.H2(f"{avg_wait_time} min", className="text-white fw-bold")
        ], className="card bg-warning text-white shadow-lg rounded p-4 col-md-3 mb-4"),
        html.Div([
            html.I(className="fas fa-dollar-sign fa-2x text-white"),
            html.H4("Cost", className="mt-2 text-white"),
            html.H2(f"${total_revenue / 1e6:.1f}M", className="text-white fw-bold")
        ], className="card bg-danger text-white shadow-lg rounded p-4 col-md-3 mb-4")
    ], className="row justify-content-center")

    top_procedures = filtered_procedures.groupby('DESCRIPTION')['BASE_COST'].sum().nlargest(10).reset_index()
    resource_fig = px.bar(
        top_procedures,
        x='DESCRIPTION',  # Categories for the x-axis
        y='BASE_COST',  # Values for the y-axis
        title='Resource Usage by Procedure',  # Title of the graph
        labels={'DESCRIPTION': 'Procedure Description', 'BASE_COST': 'Cost (in $)'},  # Axis labels
        color='DESCRIPTION',  # Color the bars based on procedure descriptions
        color_continuous_scale='Viridis',  # Color scale for differentiation
    )

    # Update the traces to make bars more readable
    resource_fig.update_traces(
        texttemplate='%{y:.2f}',  # Display the exact cost on top of each bar
        textposition='outside',  # Position the text outside the bars
        marker=dict(line=dict(width=1, color='black'))  # Add border around the bars for clarity
    )

    # Rotate X-Axis labels to 90 degrees for better readability
    resource_fig.update_layout(
        showlegend=False,  # Disable the legend if not needed
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',  # Add gridlines for the x-axis
            tickangle=45,  # Rotate the x-axis labels to 45 degrees for better readability
            title='Procedure Description',  # X-axis title
            tickmode='array',  # This helps with formatting
            tickvals=filtered_procedures['DESCRIPTION'].unique()  # Adjust the ticks
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',  # Add gridlines for the y-axis
            title='Cost (in $)'  # Y-axis title
        ),
        margin=dict(t=40, b=150, l=40, r=20),  # Increase bottom margin to avoid cutting off labels
        title_x=0.5,  # Center the title
        height=600,  # Set a larger height to make the graph more spacious
    )

    # Cost Analysis
    encounter_costs = filtered_encounters[['DESCRIPTION', 'TOTAL_CLAIM_COST']].copy()
    encounter_costs['TOTAL_CLAIM_COST'] = encounter_costs['TOTAL_CLAIM_COST'].astype(float)
    encounter_costs = encounter_costs.groupby('DESCRIPTION', as_index=False).sum()
    encounter_costs = encounter_costs.sort_values(by='TOTAL_CLAIM_COST', ascending=False)

    top_n = 7
    if len(encounter_costs) > top_n:
        top_data = encounter_costs.head(top_n)
        others_sum = encounter_costs['TOTAL_CLAIM_COST'][top_n:].sum()
        others_row = pd.DataFrame([{'DESCRIPTION': 'Others', 'TOTAL_CLAIM_COST': others_sum}])
        encounter_costs = pd.concat([top_data, others_row], ignore_index=True)

    cost_fig = px.bar(
        encounter_costs,
        x='DESCRIPTION',
        y='TOTAL_CLAIM_COST',
        color='DESCRIPTION',  # Assigns different colors to each bar
        title='Cost Analysis by Procedure',
        text='TOTAL_CLAIM_COST'
    )

    cost_fig.update_traces(
        texttemplate='$%{text:,.2f}',
        textposition='inside',  # Moves the label inside the bar
        insidetextanchor='start'  # Keeps it readable even for tall bars
    )

    cost_fig.update_layout(
        xaxis_title='Procedure',
        yaxis_title='Total Claim Cost ($)',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False  # Hides legend since color already maps to x labels
    )

    return metrics_display, resource_fig, cost_fig

# Callback for Costs and Utilization Dashboard
@app.callback(
    Output('utilization-graph', 'figure'),
    [Input('time-aggregation', 'value')]
)
def update_graph(selected_period):
    if selected_period == 'Day of Week':
        data = dow_utilization.reset_index()
        x_col = 'MEDICATION_START'
        title = "Encounters Per Day of the Week"
        xaxis_title = "Day of the Week"
        category_orders = {'MEDICATION_START': days_order}
    elif selected_period == 'Week':
        data = weekly_utilization.reset_index()
        data['week'] = data['week'].astype(str)
        x_col = 'week'
        title = "Encounter per Week"
        xaxis_title = "Week"
        category_orders = None
    else:
        data = monthly_utilization.reset_index()
        data['month'] = data['month'].astype(str)
        x_col = 'month'
        title = "Encounter per Month"
        xaxis_title = "Month"
        category_orders = None
    fig = px.bar(data, x=x_col, y=data.columns[1:], barmode='stack',
                 color_discrete_map=color_map, category_orders=category_orders)
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title="Number of Encounters", title_x=0.5)
    return fig

# Callback for Costs and Utilization Dashboard - Forecast Graph
@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('forecast-period', 'value')]
)
def update_forecast_graph(forecast_period):
    if forecast_period == 'W':
        utilization_df = weekly_utilization
        title = "Weekly Encounter Forecast"
    elif forecast_period == 'M':
        utilization_df = monthly_utilization
        title = "Monthly Encounter Forecast"
    else:
        return go.Figure()

    forecast = forecast_encounters(utilization_df, period=forecast_period)
    df = utilization_df.sum(axis=1).reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = df['ds'].dt.to_timestamp()

    fig = go.Figure()
    # Historical data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Historical', marker=dict(color='blue')))
    # Forecasted data
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='green')))
    # Confidence intervals
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty',
                   fillcolor='rgba(68, 68, 68, 0.3)', showlegend=False))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Number of Encounters',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white'
    )
    return fig
# Callback for Patient Analytics Dashboard with Total Patient Count
@app.callback(
    [Output('patients-table', 'data'),
     Output('total-patients', 'children')],  # Add the output for Total Patients
    [
        Input('search-name', 'value'),
        Input('department-filter', 'value'),
        Input('status-filter', 'value'),
        Input('gender-filter', 'value'),
        Input('days-filter', 'value'),
        Input('sort-order', 'value')
    ]
)
def update_table(search_name, department, status, gender, days, sort_order):
    # Make a copy of the dataframe to apply filters
    df = merged_analytics_df.copy()

    # Apply filters based on user input
    if search_name:
        df = df[df['FIRST'].str.contains(search_name, case=False, na=False)]
    if department:
        df = df[df['ENCOUNTERCLASS'].isin(department)]
    if status:
        df = df[df['STATUS'].isin(status)]
    if gender:
        df = df[df['GENDER'].isin(gender)]
    if days:
        cutoff_date = datetime.date.today() - datetime.timedelta(days=days)
        df = df[df['DATE'] >= cutoff_date]
    if sort_order:
        df = df.sort_values(by='DATE', ascending=(sort_order == 'asc'))

    # Remove duplicates based on Patient ID to avoid counting the same patient multiple times
    df_unique_patients = df.drop_duplicates(subset=['PATIENT'])  # Ensuring uniqueness based on Patient ID

    # Calculate the total number of unique patients after filtering
    total_patients = len(df_unique_patients)

    # Return the filtered table data and the total unique patient count
    return df.to_dict('records'), f"Total Patients: {total_patients:,}"



# Callback for Patient Information Dashboard - Patient Details and Encounter Table
@app.callback(
    [Output('patient-details', 'children'), Output('encounter-table', 'data')],
    [Input('patient-select', 'value')]
)
def update_patient_info(patient_id):
    if not patient_id:
        return [], []
    patient_details = patients_df[patients_df['Id'] == patient_id].iloc[0]
    today = datetime.date.today()
    birthdate = pd.to_datetime(patient_details['BIRTHDATE'], errors='coerce').date()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    patient_name = f"{patient_details.get('PREFIX', '')} {patient_details['FIRST']} {patient_details['LAST']} {patient_details.get('SUFFIX', '')}".strip()

    patient_info = [
        html.P([html.I(className="fas fa-id-badge text-secondary"), f" Patient ID: {patient_details['Id']}"], className="mb-2"),
        html.P([html.I(className="fas fa-user text-success"), f" Name: {patient_name}"], className="mb-2"),
        html.P([html.I(className="fas fa-ring text-warning"), f" Marital Status: {patient_details.get('MARITAL', 'N/A')}"], className="mb-2"),
        html.P([html.I(className="fas fa-palette text-info"), f" Race: {patient_details.get('RACE', 'N/A')}"], className="mb-2"),
        html.P([html.I(className="fas fa-globe text-primary"), f" Ethnicity: {patient_details.get('ETHNICITY', 'N/A')}"], className="mb-2"),
        html.P([html.I(className="fas fa-venus-mars text-purple"), f" Gender: {patient_details['GENDER']}"], className="mb-2"),
        html.P([html.I(className="fas fa-map-marker-alt text-danger"), f" Address: {patient_details.get('ADDRESS', '')}, {patient_details.get('CITY', '')}, {patient_details.get('STATE', '')} {patient_details.get('ZIP', '')}"], className="mb-2")
    ]

    encounters_person = encounters_df[encounters_df['PATIENT'] == patient_id]
    obs_grouped = observations_df[observations_df['PATIENT'] == patient_id] \
        .groupby('ENCOUNTER')['DESCRIPTION'].apply(lambda x: '; '.join(x)).reset_index()
    encounter_history = pd.merge(encounters_person, obs_grouped, left_on='Id', right_on='ENCOUNTER',
                                 how='left', suffixes=('', '_obs'))
    encounter_history['treatment'] = encounter_history['STOP'].apply(lambda x: 'Healthy' if pd.notnull(x) else '')
    encounter_history_table = encounter_history[['START', 'REASONDESCRIPTION', 'DESCRIPTION_obs', 'treatment']].copy()
    encounter_history_table.columns = ['Date', 'Reason', 'Observations', 'Treatment']
    encounter_history_table['Date'] = pd.to_datetime(encounter_history_table['Date']).dt.strftime('%b. %d, %Y')
    encounter_history_table['Observations'] = encounter_history_table['Observations'].fillna('No observations')
    return patient_info, encounter_history_table.to_dict('records')

# Callback for Patient Information Dashboard - Well-being Graph
@app.callback(
    Output("wellbeing-graph", "figure"),
    [Input("patient-select", "value"),
     Input("wellbeing-metric", "value")]
)
def update_wellbeing_graph(patient_id, metric):
    if not patient_id or not metric:
        return px.line(title="Select a patient and metric to view trends")

    obs = observations_df[
        (observations_df["PATIENT"] == patient_id) &
        (observations_df["DESCRIPTION"] == metric)
    ].copy()

    if obs.empty:
        return px.line(title=f"No {metric} data found for this patient")

    obs["DATE"] = pd.to_datetime(obs["DATE"])
    obs.sort_values("DATE", inplace=True)
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")

    if metric == "Body Weight":
        obs["VALUE"] *= 2.20462  # Convert to lbs
    if metric == "BMI":
        obs = obs[obs["VALUE"].notna()]

    obs["MONTHS"] = ((obs["DATE"] - obs["DATE"].min()).dt.days / 30.44).round(2)

    fig = px.line(
        obs,
        x="MONTHS",
        y="VALUE",
        title=f"{metric} Trends Over Time",
        labels={"MONTHS": "Months", "VALUE": metric},
        markers=True
    )
    fig.update_traces(mode="lines+markers", text=obs["VALUE"].round(1).astype(str))
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    return fig

# Callback for Patient Information Dashboard - Conditional Display
@app.callback(
    Output("patient-info-section", "children"),
    Input("patient-select", "value")
)
def display_patient_info(patient_id):
    if not patient_id:
        return []

    patient_details_card = html.Div([
        html.H3("🆔 Patient Details", className="h5 mb-3 text-secondary fw-bold"),
        html.Div([
            html.Div([
                html.Div(id="patient-photo", className="mb-3",
                         style={"width": "100px", "height": "100px", "borderRadius": "50%",
                                "backgroundColor": "#dee2e6"}),
                html.Div(id="patient-details-text")
            ])
        ], id="patient-details", className="card-body patient-info")
    ], className="card shadow-lg p-4 bg-light rounded h-100")

    encounter_table_card = html.Div([
        html.H3("📜 Encounter History", className="h5 mb-3 text-secondary fw-bold"),
        dash_table.DataTable(
            id="encounter-table",
            columns=[
                {"name": "📅 Date", "id": "Date"},
                {"name": "📝 Reason", "id": "Reason"},
                {"name": "🔬 Observations", "id": "Observations"},
                {"name": "💊 Treatment", "id": "Treatment"}
            ],
            style_table={"overflowX": "auto", "borderRadius": "12px", "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"},
            style_cell={"textAlign": "left", "padding": "10px", "fontSize": "14px"},
            style_header={"backgroundColor": "#343a40", "color": "white", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "rgb(248, 248, 248)"},
                {"if": {"row_index": "even"}, "backgroundColor": "rgb(255, 255, 255)"}
            ],
            style_as_list_view=True
        )
    ], className="card shadow-lg p-4 mb-4 bg-light rounded")

    wellbeing_graph_card = html.Div([
        html.H3("📈 Well-being Trends", className="h5 mb-3 text-secondary fw-bold"),
        html.Div([
            html.Label("📊 Select Metric", className="form-label fw-bold"),
            dcc.Dropdown(
                id="wellbeing-metric",
                options=[
                    {"label": "Weight", "value": "Body Weight"},
                    {"label": "Blood Pressure - Systolic", "value": "Systolic Blood Pressure"},
                    {"label": "Blood Pressure - Diastolic", "value": "Diastolic Blood Pressure"},
                    {"label": "BMI", "value": "BMI"}
                ],
                value="Body Weight",
                clearable=False,
                className="form-select shadow-sm mb-3"
            )
        ], className="mb-3"),
        dcc.Graph(id="wellbeing-graph")
    ], className="card shadow-lg p-4 bg-light rounded")

    return [
        html.Div([
            html.Div(patient_details_card, className="col-md-6 mb-4 d-flex align-items-stretch"),
            html.Div(wellbeing_graph_card, className="col-md-6 mb-4")
        ], className="row gx-4"),
        encounter_table_card
    ]

# Main Application Layout with Sidebar Navigation
app.layout = html.Div([
    html.Div([
        html.H1("🏥 XYZ Group of Hospitals", className="display-4 text-center mt-4 text-white fw-bold",
                style={
                    "background": "linear-gradient(to right, #0066cc, #003366)",
                    "color": "white",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "border": "3px solid #004080",
                    "boxShadow": "2px 2px 10px rgba(0, 0, 0, 0.2)",
                    "textAlign": "center"
                }),
    ], className="container-fluid header-content"),
    html.Div([
        html.Div([
            html.Ul([
                html.Li(
                    html.Button([html.I(className="fas fa-tachometer-alt"), " Home"], id='nav-overview', className='nav-link active',
                                style={'backgroundColor': 'transparent', 'color': 'white', 'border': 'none',
                                       'outline': 'none', 'fontSize': '18px', 'cursor': 'pointer',
                                       'padding': '15px'}),
                    className='nav-item', style={'marginBottom': '15px'}
                ),
                html.Li(
                    html.Button([html.I(className="fas fa-user-injured"), " Encounters "], id='nav-costs', className='nav-link',
                                style={'backgroundColor': 'transparent', 'color': 'white', 'border': 'none',
                                       'outline': 'none', 'fontSize': '18px', 'cursor': 'pointer',
                                       'padding': '15px'}),
                    className='nav-item', style={'marginBottom': '15px'}
                ),
                html.Li(
                    html.Button([html.I(className="fas fa-chart-bar"), " Patient Analytics"], id='nav-analytics', className='nav-link',
                                style={'backgroundColor': 'transparent', 'color': 'white', 'border': 'none',
                                       'outline': 'none', 'fontSize': '18px', 'cursor': 'pointer',
                                       'padding': '15px'}),
                    className='nav-item', style={'marginBottom': '15px'}
                ),
                html.Li(
                    html.Button([html.I(className="fas fa-user-md"), " Patient Information"], id='nav-info', className='nav-link',
                                style={'backgroundColor': 'transparent', 'color': 'white', 'border': 'none',
                                       'outline': 'none', 'fontSize': '18px', 'cursor': 'pointer',
                                       'padding': '15px'}),
                    className='nav-item', style={'marginBottom': '15px'}
                )
            ], className='nav flex-column', style={'listStyleType': 'none', 'padding': '0'})
        ], className='col-md-2 sidebar',
            style={'minHeight': '100vh', 'background': 'linear-gradient(135deg, #667eea, #764ba2)',
                   'color': 'white', 'padding': '30px', 'borderRadius': '0 8px 8px 0',
                   'boxShadow': '2px 0 10px rgba(0,0,0,0.3)'}),
        html.Div([
            html.Div(overview_layout, id='content-overview', style={'display': 'block'}),
            html.Div(costs_layout, id='content-costs', style={'display': 'none'}),
            html.Div(analytics_layout, id='content-analytics', style={'display': 'none'}),
            html.Div(info_layout, id='content-info', style={'display': 'none'})
        ], className='col-md-10 p-4 content-area')
    ], className='row no-gutters')
], className="container-fluid", style={'fontFamily': 'Roboto, sans-serif'})

# Callback to Toggle Dashboard Visibility
@app.callback(
    [Output('content-overview', 'style'), Output('content-costs', 'style'),
     Output('content-analytics', 'style'), Output('content-info', 'style'),
     Output('nav-overview', 'className'), Output('nav-costs', 'className'),
     Output('nav-analytics', 'className'), Output('nav-info', 'className')],
    [Input('nav-overview', 'n_clicks'), Input('nav-costs', 'n_clicks'),
     Input('nav-analytics', 'n_clicks'), Input('nav-info', 'n_clicks')]
)
def toggle_content(n_overview, n_costs, n_analytics, n_info):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
            'nav-link active', 'nav-link', 'nav-link', 'nav-link'
        )
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'nav-overview':
        return (
            {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
            'nav-link active', 'nav-link', 'nav-link', 'nav-link'
        )
    elif button_id == 'nav-costs':
        return (
            {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'},
            'nav-link', 'nav-link active', 'nav-link', 'nav-link'
        )
    elif button_id == 'nav-analytics':
        return (
            {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'},
            'nav-link', 'nav-link', 'nav-link active', 'nav-link'
        )
    elif button_id == 'nav-info':
        return (
            {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'},
            'nav-link', 'nav-link', 'nav-link', 'nav-link active'
        )
    return (
        {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'},
        'nav-link active', 'nav-link', 'nav-link', 'nav-link'
    )

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run(debug=True)