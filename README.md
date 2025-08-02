Data Collection and wrangling
import pandas as pd

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Basic cleaning
df.dropna(inplace=True)  # Remove nulls
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower()

# Convert data types
df['date'] = pd.to_datetime(df['date_column'])

# Save cleaned version
df.to_csv('data/cleaned_data.csv', index=False)# Applied-Data-Science-Capstone-Final-Assignment-
Applied Data Science Capstone 


EDA and Interactive Visual Analysis
import seaborn as sns
import matplotlib.pyplot as plt

# Histograms
df.hist(bins=30, figsize=(15,10))
plt.tight_layout()
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
import plotly.express as px

fig = px.scatter(df, x='feature1', y='feature2', color='category')
fig.show()

Predictive analysis methodology 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)

EDA with Visualization Results 
sns.boxplot(data=df, x='category', y='value')
plt.title("Boxplot of Value by Category")
plt.show()

px.bar(df, x='category', y='sales', title="Sales by Category").show()

EDA with SQL Results 
import sqlite3

conn = sqlite3.connect('data/my_database.db')
query = "SELECT category, AVG(sales) as avg_sales FROM sales_data GROUP BY category"
sql_df = pd.read_sql_query(query, conn)

print(sql_df)

Folium Interactive Map
import folium

# Assume df has 'latitude' and 'longitude' columns
map_ = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)

for _, row in df.iterrows():
    folium.Marker(location=[row['latitude'], row['longitude']],
                  popup=row['location_name']).add_to(map_)

map_.save('folium_map.html')

Plotly Dash Dashboard 
import dash
from dash import html, dcc
import plotly.express as px

app = dash.Dash(__name__)

fig = px.histogram(df, x='feature1')

app.layout = html.Div([
    html.H1("Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
 
 Predictive analysis (classification) results 
 from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
    

