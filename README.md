# Urban-Traffic-Flow-and-Congestion-Analysis-Using-Vehicle-Count-and-Speed-Data

Project Overview

This project focuses on preprocessing and preparing a real-world style traffic dataset for further data analysis and machine learning tasks. The dataset contains traffic-related attributes such as vehicle count, average speed, flow rate, time of day, and waiting time. Raw traffic data often contains inconsistencies, missing values, duplicate records, and scale differences — this project cleans and transforms the dataset into a model-ready format.

The goal is to build a clean, structured dataset that can be used for traffic pattern analysis, congestion prediction, and smart city transportation insights.


Dataset Features

The dataset includes the following traffic parameters:

* Vehicle Count — number of vehicles detected
* Average Speed — mean vehicle speed
* Flow Rate — vehicles passing per unit time
* Time of Day — morning / afternoon / evening / night
* Waiting Time — average signal waiting duration
* Record Index Column (removed during preprocessing)



 Preprocessing Steps Performed

The following preprocessing pipeline was implemented using Python (Pandas + Scikit-learn):

* Removed unnecessary index column (`Unnamed: 0`)
* Cleaned and standardized column names
* Removed duplicate records
* Handled missing values

  * Numeric columns → filled with median
  * Categorical columns → filled with mode
* Detected and capped outliers using IQR method
* Encoded categorical features using Label Encoding
* Standardized numeric features using StandardScaler
* Generated a cleaned dataset file for downstream tasks



Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* VS Code


How to Run

1. Place dataset and preprocessing script in same folder

```
traffic_dataset.csv
preprocess_dataset_fixed.py
```

2. Install dependencies

```
pip install pandas numpy scikit-learn
```

3. Run preprocessing

```
python preprocess_dataset_fixed.py
```

4. Output file generated:

```
traffic_dataset_cleaned.csv
Outcome

The raw traffic dataset was successfully transformed into a clean, normalized, and encoded dataset suitable for:

* Traffic congestion analysis
* Visualization dashboards
* Machine learning models
* Smart traffic prediction systems

---

Future Work

* Traffic congestion prediction model
* Visualization dashboard
* Peak-hour detection
* Smart signal timing optimization
* ML-based incident prediction

---
