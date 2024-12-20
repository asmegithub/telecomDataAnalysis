## Telecom data analysis 
# Telecommunication Dataset Analysis

This project analyzes a telecommunication dataset for the potential acquisition of TellCo, a mobile service provider in the Republic of Pefkakia. The analysis focuses on identifying user behavior patterns, insights for growth and profitability, and making a recommendation on whether TellCo is worth buying or selling.

## Overview

The dataset contains detailed records of user sessions, including metrics such as session duration, data usage, and application-specific data (e.g., social media, video streaming, etc.). The analysis aims to provide actionable insights for investment decisions and marketing strategies.

## Project Structure

```
project-root/
├── db/              # database connections
├── scripts/           # Python scripts for data analysis and visualization
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies
```

## Tasks

### Task 1.1: User Behavior Overview
Aggregate user data to provide the following metrics:
- Number of xDR sessions per user
- Total session duration per user
- Total upload (UL) and download (DL) data per user
- Total data volume (Bytes) per user

### Exploratory Data Analysis (EDA)

#### Univariate Analysis:
- Histograms and box plots of key metrics like `Dur. (ms)` (session duration) and `Total Data (Bytes)` to understand distributions and detect outliers.

#### Bivariate Analysis:
- Scatter plots to explore relationships between key variables, such as session duration and total data volume.
- Example insight: Users with longer sessions tend to have higher data usage, though variability exists.

### Outlier Detection
- Investigate sessions with unusually high duration or data usage for anomalies.

### Marketing Insights
- Segment users based on behavior metrics (e.g., heavy vs. light data users) to recommend personalized offers.

## Key Findings
1. **User Behavior:**
   - Most sessions are short, but data usage varies significantly.
   - Outliers with high durations or data volumes may indicate heavy users or anomalies.

2. **Correlation:**
   - Total data usage increases with session duration but is not strictly linear.

3. **Marketing Opportunities:**
   - Target heavy data users for premium plans.
   - Address user segments with personalized promotions.

## Usage

### Prerequisites
- Python 3.10 or higher
- Required libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:asmegithub/telecomDataAnalysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis
1. Place the dataset in the `data/` directory. or create a connection request at `db/`
2. Execute analysis scripts from the `scripts/` folder:
   ```bash
   python scripts/analysis.py
   ```
3. View generated plots and outputs in the `outputs/` directory.

## Deliverables
1. **Insights Report:**
   - Summary of user behavior and marketing recommendations.
2. **Dashboard (if applicable):**
   - Interactive visualizations for easy interpretation.
3. **Codebase:**
   - Modular and reusable scripts for analysis.

## Next Steps
- Further refine segmentation models to enhance marketing strategies.
- Investigate outliers to determine their impact on business decisions.

## Contributors
- **Asmare Zelalem** - Junior Full Stack Developer and Data Analyst

## Contact
For questions or suggestions, please contact Asmare at asmare.zelalem@aau.edu.et.
