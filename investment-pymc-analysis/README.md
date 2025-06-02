# Investment Decision Analysis with PyMC

This project aims to analyze investment decisions using Bayesian modeling with PyMC. The analysis focuses on understanding the factors influencing investment outcomes based on a dataset of investment decisions.

## Project Structure

- **data/**: Contains the dataset used for analysis.
  - `investment_decisions.csv`: This file includes columns for trial number, response time, investment outcome, probabilities, gains, and losses.

- **src/**: Contains the source code for the analysis.
  - `model.py`: Implements the PyMC model for analyzing investment decisions, defining the Bernoulli probability function for binary outcomes and incorporating informative priors.
  - `preprocess.py`: Includes functions for loading and preprocessing the investment data, ensuring it is in the correct format for analysis.
  - `utils.py`: Contains utility functions for data visualization and model diagnostics.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis.
  - `exploratory_analysis.ipynb`: Used for visualizations and insights into the investment decision data before modeling.

- **requirements.txt**: Lists the necessary Python packages required for the project, including PyMC and other dependencies.

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd investment-pymc-analysis
pip install -r requirements.txt
```

## Usage

1. **Preprocess the Data**: Use the functions in `src/preprocess.py` to load and prepare the investment data.
2. **Run the Model**: Execute the PyMC model defined in `src/model.py` to analyze the investment decisions.
3. **Explore the Results**: Use the Jupyter notebook `notebooks/exploratory_analysis.ipynb` for visualizations and insights.

## Interpretation of Results

The results from the PyMC model will provide insights into the factors influencing investment decisions, including the estimated probabilities of investment outcomes based on the specified parameters. The analysis can help in understanding the decision-making process and improving investment strategies.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.