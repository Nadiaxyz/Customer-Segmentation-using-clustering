# Customer-Segmentation-using-clustering
# Customer Segmentation using K-Means Clustering

## Project Overview
This project applies K-Means clustering to segment customers based on their purchasing behavior. The dataset used is `Mall_Customers.csv`, which contains customer details such as age, annual income, and spending score.

## Dataset
- **File**: `Mall_Customers.csv`
- **Features Used**:
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

## Project Structure
```
Customer-Segmentation/
│-- customer_segmentation.ipynb  # Jupyter Notebook with complete analysis
│-- Mall_Customers.csv           # Dataset
│-- README.md                    # Project Documentation
```

## Steps to Run the Project
### 1. Install Required Libraries
Ensure you have the necessary Python libraries installed. Run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Jupyter Notebook
Open the notebook and execute the cells step by step:
```bash
jupyter notebook customer_segmentation.ipynb
```

## Methodology
### Step 1: Load the Dataset
- Read `Mall_Customers.csv` using pandas.
- Perform basic checks and clean column names.

### Step 2: Exploratory Data Analysis (EDA)
- Display dataset summary.
- Handle missing values (if any).
- Visualize the distributions of Age, Annual Income, and Spending Score.

### Step 3: Data Preprocessing
- Extract relevant features.
- Standardize the dataset using `StandardScaler`.

### Step 4: Determine Optimal Clusters
- Use the **Elbow Method** to find the best number of clusters.
- Plot Within-Cluster Sum of Squares (WCSS) for different values of K.

### Step 5: Apply K-Means Clustering
- Train K-Means with the optimal number of clusters.
- Assign cluster labels to customers.

### Step 6: Visualize Clusters
- Use scatter plots to visualize customer segments based on `Annual Income` and `Spending Score`.

### Step 7: Interpret the Clusters
- Compute summary statistics for each cluster.
- Analyze the customer segments based on spending behavior.

## Results
- Identified distinct customer segments based on income and spending habits.
- Provided insights into customer groups that businesses can target for marketing strategies.

## Author
Nadia Imtiaz

## License
This project is open-source and available under the MIT License.

