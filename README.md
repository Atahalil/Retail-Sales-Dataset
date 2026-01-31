# Retail-Sales-Dataset
# Retail Sales Data Analysis (Python)

This project performs an end-to-end exploratory data analysis (EDA) on a retail sales dataset to extract business insights related to customer behavior, sales performance, and product categories.

---

## Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## Dataset
- Source: Kaggle (public retail sales dataset)
- Format: CSV  
- The dataset is not included in the repository and should be downloaded separately.

---

## Data Processing
- CSV ingestion and column name standardization  
- Datetime parsing (`Date`)  
- Numeric type coercion (`Age`, `Quantity`, `Price per Unit`, `Total Amount`)  
- Missing value handling (key fields filtered, categorical imputation)  
- Duplicate removal based on `Transaction ID`  
- Outlier detection using the IQR method (flagging only)

---

## Analysis Performed
- Descriptive statistics for numeric variables  
- Product category–level sales aggregation  
- Customer segmentation by gender and age groups  
- Financial KPIs (total revenue, average transaction value, dispersion metrics)  
- Time-based analysis with monthly sales aggregation  
- Pearson correlation analysis on numeric features  

---

## Visualizations
- Total sales by product category  
- Monthly total sales trend  
- Monthly spending trends by age group  
- Correlation heatmap  

All plots are exported to the `plots/` directory.

---

## Outputs
- Cleaned dataset: `plots/cleaned_retail_sales.csv`  
- Visualization files: `plots/*.png`

---


## Key Insights
- Certain product categories contribute disproportionately to total revenue.
- Middle-aged customer segments show higher average transaction values.
- Monthly sales exhibit clear seasonal patterns.

---

## Purpose
This project demonstrates practical skills in **data cleaning, exploratory data analysis, statistical reasoning, and business-oriented data visualization** using real-world retail data.  
Suitable as a **data analyst / data science portfolio project**.



