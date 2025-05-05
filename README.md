# AI-Powered Data Analysis Agent for Commuting Times

**Author:** Daniil Vladimirov  
**Student Number:** 3154227  
**Module:** Analysis of Big Data  

## Overview

This project presents an automated data analysis pipeline built using Python, LlamaIndex, and AI agents. It analyzes the Dublin Canal Cordon commuting times dataset, performing comprehensive data quality assessment, cleaning, exploratory data analysis (EDA), statistical analysis, regression modeling, and reporting.

The primary goal is to demonstrate an advanced, reproducible, and intelligent approach to data analysis, showcasing capabilities beyond traditional statistical software like JASP, which was initially suggested for the assignment. The workflow leverages Large Language Models (LLMs) via LlamaIndex to automate steps, suggest cleaning strategies, perform analysis, and generate narrative reports, while incorporating human oversight for critical decisions.

## Dataset

The analysis focuses on the `Commute_Times_V1.csv` dataset, containing 281 records of commuter interviews within Dublin's canal cordon. Key variables include:
- `Case`: Unique identifier.
- `Mode`: Transportation mode (Walk, Cycle, Bus, Car).
- `Distance`: Straight-line distance from home to work (km).
- `Time`: Actual travel time (minutes).

## Features

*   **Automated Workflow:** End-to-end pipeline using LlamaIndex's event-driven workflow framework.
*   **Data Quality Assessment:** Comprehensive checks for missing values, duplicates, outliers (IQR, Z-score), impossible values, and data type consistency, generating a quality score (initial score: 94.48/100).
*   **AI-Assisted Cleaning:** An AI agent analyzes the quality report and suggests cleaning steps with statistical justifications.
*   **Human-in-the-Loop:** Interactive consultation step (`consultation.py`) allows user review and approval/modification of AI-suggested cleaning actions.
*   **Systematic Cleaning:** Implements approved cleaning steps (standardization, imputation, outlier capping, duplicate removal) using a dedicated `DataCleaner` class.
*   **Exploratory Data Analysis (EDA):** Generates descriptive statistics, distribution plots (histograms, density plots), and mode-specific analysis (boxplots, violin plots).
*   **Advanced Statistical Analysis:** Calculates advanced statistics (skewness, kurtosis), confidence intervals, performs normality tests (Shapiro-Wilk), and significance tests (ANOVA, Tukey HSD).
*   **Regression Modeling:**
    *   Fits linear regression models for the full dataset and each transportation mode separately.
    *   Performs in-depth comparison between the full dataset model and the highly predictive Cycle-specific model (R² = 0.9791 vs 0.4282).
    *   Includes model validation (residual analysis, cross-validation).
*   **Advanced Modeling:** Explores and compares alternative model forms (Polynomial Regression, Logarithmic Transformations) using AIC/BIC for model selection, identifying a Log-Log model as the best fit for the full dataset (R² = 0.6144).
*   **Automated Visualization:** Generates various plots (scatter, line, bar, heatmap, Q-Q plots, etc.) saved to the `plots/` directory.
*   **Comprehensive Reporting:** Generates detailed JSON reports for data quality, cleaning, statistical analysis, model validation, and predictions, saved to the `reports/` directory. An AI agent also synthesizes findings into a final Markdown report.

## Architecture

The project utilizes LlamaIndex's `Workflow` class to orchestrate a series of steps triggered by events. Specialized AI agents (`FunctionCallingAgent`) handle specific tasks:

1. **Data Preparation Agent:** Analyzes initial data quality and statistics to propose a cleaning plan.
2. **Data Analysis Agent:** Executes Python/pandas code (via `FunctionTool`) to implement cleaning, perform analysis, and save results based on instructions.

A `Context` object maintains state (DataFrame, reports, etc.) throughout the workflow. The `PandasQueryEngine` allows agents to interact with the DataFrame using natural language queries, although the primary analysis relies on structured Python functions.

**Workflow Steps:**

1. **Setup:** Loads the dataset, creates a PandasQueryEngine, performs quality assessment
2. **Data Preparation:** AI agent analyzes quality assessment and suggests cleaning steps
3. **Human Consultation:** Interactive step for user review of AI cleaning suggestions
4. **Data Modification:** Applies approved cleaning operations using DataCleaner class
5. **Advanced Statistical Analysis:** Computes statistics, confidence intervals, normality tests
6. **Regression Modeling:** Fits and validates linear regression models and advanced alternatives 
7. **Analysis Reporting:** Synthesizes findings into comprehensive reports
8. **Visualization Creation:** Generates standard and advanced visualizations
9. **Report Finalization:** Verifies all reports and regenerates any missing ones

The event-driven architecture allows for clean separation of concerns, error isolation, and extensibility. Events pass data between steps, creating a clear flow of information through the analysis pipeline.

## Installation

```bash
# Clone the repository
git clone https://github.com/username/data_analysis_agent.git
cd data_analysis_agent

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create an .env file and add your LLM API key (e.g., OpenAI)
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Usage

To run the analysis on a CSV dataset:

```bash
python run.py path/to/your/dataset.csv
```

For the provided demonstration dataset:

```bash
python run.py Commute_Times_V1.csv
```

During execution, you'll be prompted to review and approve AI-suggested cleaning steps. Follow the numbered prompts to select which cleaning operations to perform.

## Directory Structure

```
data_analysis_agent/           # Main project directory
├── run.py                     # Entry point for running the analysis
├── workflow.py                # Core workflow implementation with event-driven architecture
├── agents.py                  # AI agents definitions and configuration
├── events.py                  # Event classes for workflow communication
├── data_quality.py            # Quality assessment and data cleaning implementations
├── consultation.py            # Human-in-the-loop consultation implementation
├── advanced_analysis.py       # Advanced statistical analysis functions
├── regression_analysis.py     # Regression modeling and comparison
├── model_validation.py        # Model validation and assumption checking
├── visualization.py           # Data visualization generation
├── reporting.py               # Report generation and aggregation
├── tools/                     # Custom tools for AI agents
│   ├── execute_pd_tool.py     # Tool for executing pandas code
│   └── save_dataframe_tool.py # Tool for saving DataFrames
├── plots/                     # Generated visualizations
│   ├── advanced/              # Advanced statistical plots
│   ├── cleaning_comparisons/  # Before/after cleaning visualizations
│   ├── models/                # Model comparison plots
│   ├── predictions/           # Prediction visualization
│   ├── regression/            # Linear regression plots
│   └── validation/            # Model validation plots
└── reports/                   # Generated JSON reports
    ├── data_quality_report.json
    ├── cleaning_report.json
    ├── statistical_analysis_report.json
    ├── regression_models.json
    ├── advanced_models.json
    └── prediction_results.json
```

## Key Results

The analysis revealed several key insights about commuting in Dublin:

1. **Mode-Specific Differences:** Significant variation in commute times across different transport modes (ANOVA p < 0.0001)
2. **Predictive Power:** The Cycle-specific model achieves remarkably high predictive accuracy (R² = 0.9791) compared to the full dataset model (R² = 0.4282)
3. **Best Model Form:** A log-log transformation model (R² = 0.6144) outperforms the linear model for the full dataset
4. **Quality Improvements:** Data cleaning improved model quality by standardizing values, handling outliers, and addressing missing data

## Advantages Over Traditional Tools

This automated approach demonstrates several advantages over traditional statistical tools like JASP:

1. **Reproducibility:** Complete workflow automation with code-based implementation
2. **Advanced Cleaning:** Systematic data cleaning with detailed before/after metrics
3. **Rich Statistical Analysis:** Advanced metrics including skewness, kurtosis, confidence intervals
4. **Mode-Specific Analysis:** Automatic data segmentation revealing critical differences
5. **Advanced Modeling:** Exploration of multiple model forms with robust comparison
6. **Comprehensive Validation:** Cross-validation, residual analysis, and performance metrics
7. **Advanced Visualization:** Rich set of visualizations revealing patterns not apparent from numerical summaries

## Requirements

The project requires the following key dependencies:
- llama-index>=0.10.1 and related components
- pandas>=2.0.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- scikit-learn>=1.0.0
- statsmodels>=0.13.0

See `requirements.txt` for the complete list of dependencies.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

This project is available for academic and educational purposes.

## Acknowledgments

- Dublin City Council for the original dataset
- LlamaIndex for the workflow and agent framework
- [Insert additional acknowledgments as appropriate]

