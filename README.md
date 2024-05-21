
# ElasticNet Regression with Cross-Validation (Coordinate Descent)

This project implements ElasticNet regression using coordinate descent with cross-validation for predicting a response variable from various features. The algorithm performs cross-validation to select the best hyperparameters and includes features like parameter tuning and coefficient plotting.

## Features

- **ElasticNet Regression:** Combines L1 and L2 penalties for regression.
- **Coordinate Descent Optimization:** Uses coordinate descent for parameter optimization.
- **Cross-Validation:** Uses cross-validation to select the best hyperparameters.
- **Standardization:** Standardizes the design matrix for better performance.
- **Visualization:** Plots the effect of tuning parameters on coefficients and cross-validation error.

## Requirements

- Python 3.6+
- numpy
- matplotlib
- openpyxl

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/JustinRudisal/elasticnet-regression-coordinate-descent.git
   cd elasticnet-regression-coordinate-descent
   ```

2. **Install dependencies:**

   ```bash
   pip install numpy matplotlib openpyxl
   ```

3. **Run the script:**

   ```bash
   python elasticnet_regression_cd.py
   ```

4. **Follow the prompts to specify the file path for the input data:**

   ```
   Please specify the filepath of the input file: 
   ```

5. **View the output:**

   The script will display the best alpha and lambda values, plot the effect of tuning parameters, and save coefficients to an Excel file.

## Code Overview

- **coordinate_descent_elastic_net:** Performs coordinate descent for ElasticNet regression.
- **cross_validation:** Implements cross-validation logic.
- **center_and_standardize:** Centers and standardizes the design matrix and response vector.
- **retrain:** Retrains the model with the best parameters and prints the coefficients.
- **read_in_file:** Reads and processes input data.
- **plot_effect_of_the_lambda_value_on_the_inferred_ridge_regression_coefficients:** Plots the effect of lambda values on regression coefficients.
- **plot_effect_of_lambda_value_on_cross_validation_error:** Plots the effect of lambda values on cross-validation error.
- **write_excel_data:** Writes data to an Excel file.
- **adjust_column_widths:** Adjusts column widths in Excel for better readability.

## Customization

You can customize various parameters in the script to suit your needs:

- **LAMBDA_VALUES:** List of lambda values for cross-validation.
- **NUMBER_OF_FOLDS:** Number of folds for cross-validation.
- **MAX_ITERATIONS:** Maximum number of iterations for coordinate descent.
- **CONVERGENCE_TOLERANCE:** Tolerance for convergence in coordinate descent.
- **ALPHAS:** List of alpha values for ElasticNet.

## Example

Here is an example of running the script:

```plaintext
Please specify the filepath of the input file: 

Best Alpha: 0.5
Best Lambda Value: 0.1

Parameters from retraining in readable terms:
Income: -273.9703
Limit: 419.8016
Rating: 195.0083
Cards: 23.4947
Age: -11.0124
Education: -3.3392
Gender: -5.1786
Student: 127.7056
Married: -3.6001

Predictions saved to Deliverable 4 Predictions - Justin Rudisal.xlsx
```

## Acknowledgments

- This project was created as part of an assignment for CAP 5625 at Florida Atlantic University.
