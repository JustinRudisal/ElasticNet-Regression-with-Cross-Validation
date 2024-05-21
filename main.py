# Author: Justin Rudisal
# Assignment 3 CAP 5625
import csv
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook

NAMES = ["Income", "Limit", "Rating", "Cards","Age", "Education", "Gender", "Student", "Married"]

LAMBDA_VALUES = [10**exp for exp in range(-2, 7)] 
NUMBER_OF_FOLDS = 5
MAX_ITERATIONS = 1000
CONVERGENCE_TOLERANCE = 1e-6
ALPHAS = [0, 0.2, 0.4, 0.6, 0.8, 1]

def coordinate_descent_elastic_net(design_matrix, response_vector, lambda_value, alpha):
    number_of_features = design_matrix.shape[1]

    # Step 3: Precompute
    b_k = np.sum(design_matrix ** 2, axis=0)

    # Step 4: Randomly initialize the parameter vector
    parameter_vector = np.random.uniform(-1, 1, number_of_features)
    previous_parameter_vector = np.copy(parameter_vector)

    # Step 6: Repeat Step 5 for 1000 iterations or until convergence
    for iteration in range(MAX_ITERATIONS):
        for current_feature_index in range(number_of_features):
            
            # Step 5: The main bulk of the algorithm
            a_k = np.dot(design_matrix[:, current_feature_index].T, (response_vector - np.dot(design_matrix, parameter_vector) + (design_matrix[:, current_feature_index] * parameter_vector[current_feature_index])))
            numerator = abs(a_k) - (lambda_value * (1 - alpha) / 2)
            if numerator > 0:
                parameter_vector[current_feature_index] =  (np.sign(a_k) * numerator) / (b_k[current_feature_index] + (lambda_value * alpha))
            else:
                parameter_vector[current_feature_index] = 0

        # Checking for that "or until convergence" part
        convergence_detection = np.abs(parameter_vector - previous_parameter_vector)
        if np.all(convergence_detection < CONVERGENCE_TOLERANCE):
            break

        previous_parameter_vector = np.copy(parameter_vector)

    return parameter_vector


def cross_validation(design_matrix, response_vector, lambda_values, number_of_folds):
    print("Starting cross validation.")
    number_of_observations = design_matrix.shape[0]
    cross_validation_error_values = {}
    parameter_vector_coefficients = {}

    data_in_each_fold = np.arange(number_of_observations) % number_of_folds

    for alpha in ALPHAS:
        print(f"Cross validating for alpha {alpha}.")
        for lambda_value in lambda_values:
            print(f"Cross validating for lambda value {lambda_value}.")
            mean_squared_error_per_fold = []
            parameter_vectors_per_fold = []

            for fold_number in range(number_of_folds):
                # Set up the training and validation data 
                training_design_matrix = design_matrix[data_in_each_fold != fold_number]
                training_response_vector = response_vector[data_in_each_fold != fold_number]
                validation_design_matrix = design_matrix[data_in_each_fold == fold_number]
                validation_response_vector = response_vector[data_in_each_fold == fold_number]

                # Center and standardize the training data
                centered_training_response_vector, standardized_training_design_matrix, centering_values, standardization_values = center_and_standardize(training_design_matrix, training_response_vector)

                # Apply the same centering and standardization values to the validation data
                validation_design_matrix = (validation_design_matrix - centering_values) / standardization_values
                validation_response_vector = validation_response_vector - np.mean(training_response_vector)

                # Do algorithm 1 (similar concept to our last assignment)
                parameter_vector = coordinate_descent_elastic_net(standardized_training_design_matrix, centered_training_response_vector, lambda_value, alpha)
                parameter_vectors_per_fold.append(parameter_vector)

                # MSE calculation for the current fold
                predicted_response_vector = np.dot(validation_design_matrix, parameter_vector)
                error_vector = validation_response_vector - predicted_response_vector
                mean_squared_error = np.mean(error_vector ** 2)
                mean_squared_error_per_fold.append(mean_squared_error)

            # Average MSE and parameter vectors across folds
            cross_validation_error = np.mean(mean_squared_error_per_fold)
            parameter_vectors = np.mean(parameter_vectors_per_fold, axis=0)

            # Store MSE and parameter vectors to use in the plots later on
            cross_validation_error_values[(alpha, lambda_value)] = cross_validation_error
            parameter_vector_coefficients[(alpha, lambda_value)] = parameter_vectors

    return cross_validation_error_values, parameter_vector_coefficients


def center_and_standardize(design_matrix, response_vector):
    centering_values = np.mean(design_matrix, axis=0)
    standardization_values = np.std(design_matrix, axis=0)

    # Handle the zero standard deviation situation that was giving me runtime warnings and causing issues :(
    standardization_values[standardization_values == 0] = 1

    # Center the training response vector
    response_vector_mean = np.mean(response_vector)
    centered_response_vector = response_vector - response_vector_mean

    # Standardize the design matrix 
    standardized_design_matrix = (design_matrix - centering_values) / standardization_values
    return centered_response_vector, standardized_design_matrix, centering_values, standardization_values


def retrain(design_matrix, response_vector, best_lambda_value, best_alpha):

    # Center and standardize
    centered_response_vector, standardized_design_matrix, centering_values, standardization_values = center_and_standardize(design_matrix, response_vector)
    
    # Run the algorithm 1 again on it
    retrained_parameter_vector = coordinate_descent_elastic_net(standardized_design_matrix, centered_response_vector, best_lambda_value, best_alpha)
    
    print("Parameters from retraining as a list: " + str(retrained_parameter_vector))
    print("Parameters from retraining in readable terms:")
    for index, coefficient in enumerate(retrained_parameter_vector):
        print(f"{NAMES[index]}: {coefficient:.4f}")
    print(f"Best Alpha: {best_alpha}, Best Lambda Value: {best_lambda_value}")


def read_in_file():
    print("Reading in the input file.")
    filepath = input("Please specify the filepath of the input file: ")

    with open(filepath, "r") as file:
        reader = csv.reader(file)
        next(reader)
        data = list(reader)

    data = np.array(data, dtype=object)

    # Handle words that can be converted to a binary 0 or 1 value
    gender_column = np.where(data[:, 6] == "Female", 1, 0).reshape(-1, 1)
    student_column = np.where(data[:, 7] == "Yes", 1, 0).reshape(-1, 1)
    married_column = np.where(data[:, 8] == "Yes", 1, 0).reshape(-1, 1)

    file_data = np.hstack((data[:, :6], gender_column, student_column, married_column, data[:, 9:])).astype(float)

    return (file_data[:, :-1], file_data[:, -1])


def plot_effect_of_the_lambda_value_on_the_inferred_ridge_regression_coefficients(design_matrix, alphas, lambda_values, parameter_vector_coefficients):
    for alpha in alphas:
        plt.figure(figsize=(10, 6))
        for idx in range(design_matrix.shape[1]): 
            plt.plot(np.log10(lambda_values),
                     [parameter_vector_coefficients[(alpha, lv)][idx] for lv in lambda_values],
                     label=NAMES[idx])
        plt.xticks(ticks=np.log10(lambda_values), labels=[f'{lv:.2f}' for lv in np.log10(lambda_values)])
        plt.xlabel("Log10 of Lambda Value")
        plt.ylabel("Inferred Coefficients")
        plt.title(f"Effect of Lambda Value on Coefficients (alpha={alpha})")
        plt.legend()
        plt.grid(True)
        plt.show()
    write_inferred_ridge_regression_coefficients_plots_to_excel(alphas, lambda_values, parameter_vector_coefficients, "Deliverable 1 Plots - Justin Rudisal.xlsx")


def plot_effect_of_lambda_value_on_cross_validation_error(alphas, lambda_values, cross_validation_error_values):
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        mse_values = [cross_validation_error_values[(alpha, lv)] for lv in lambda_values]
        plt.plot(np.log10(lambda_values), mse_values, marker="o", linestyle="-", label=f'alpha={alpha}')
    plt.xticks(ticks=np.log10(lambda_values), labels=[f'{lv:.2f}' for lv in np.log10(lambda_values)])
    plt.xlabel("Log10 of Lambda Value")
    plt.ylabel("Cross-Validation Error")
    plt.title("Lambda Value Effect on Cross Validation Error")
    plt.legend()
    plt.grid(True)
    plt.show()
    write_cve_plots_to_excel(alphas, lambda_values, cross_validation_error_values, "Deliverable 2 Plots - Justin Rudisal.xlsx")


def write_cve_plots_to_excel(alphas, lambda_values, plots, excel_filename):  
    header = ["Lambda Values (X-Axis)", "Cross-Validation Error Values (Y-Axis)"]
    write_excel_data(alphas, lambda_values, header, plots, excel_filename)
    

def write_inferred_ridge_regression_coefficients_plots_to_excel(alphas, lambda_values, plots, excel_filename):
    header = ["Lambda Values (X-Axis)"] + [name + " (Y-Axis)" for name in NAMES]
    write_excel_data(alphas, lambda_values, header, plots, excel_filename)


def write_excel_data(alphas, lambda_values, header, plots, excel_filename):
    """
    I had to do some python magic here to make a mehtod generic enough for both single value plots (such as the
    cross-validation errors) as well as lists of lists (such as the parameter vectors)
    """
    workbook = Workbook()
    active_sheet = True
    for alpha in alphas:
        if active_sheet: 
            worksheet = workbook.active
            worksheet.title = f"Alpha {alpha}"
            active_sheet = False
        else:
            worksheet = workbook.create_sheet(title=f"Alpha {alpha}")
        worksheet.append(header)
        for lambda_value in lambda_values:
            parameter_values = plots[(alpha, lambda_value)]
            if isinstance(parameter_values, np.ndarray):
                parameter_values = parameter_values.tolist()  
            elif isinstance(parameter_values, float):
                parameter_values = [parameter_values] 
            row = [lambda_value] + parameter_values
            worksheet.append(row)
        adjust_column_widths(worksheet)
    workbook.save(excel_filename)


def adjust_column_widths(worksheet):
    """
    I'm a bit OCD when it comes to code I create and the outputs it generates... and it really annoyed me
    that the excel columns weren't autoadjusting their width because of the header names. This method fixes that.
    """
    for column in worksheet.columns:
        max_length = 0
        column = [cell for cell in column]
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        width = (max_length + 2)
        worksheet.column_dimensions[cell.column_letter].width = width


def main():
    # Reading in the file to get the data and then setting up the design matrix
    design_matrix, response_vector = read_in_file()

    # Cross validation logic
    cross_validation_error_values, parameter_vector_coefficients = cross_validation(design_matrix, response_vector, LAMBDA_VALUES, NUMBER_OF_FOLDS)

    # Find which alpha and lambda values gave the smallest mean squared error during cross validation
    best_alpha, best_lambda_value = min(cross_validation_error_values, key=cross_validation_error_values.get)

    # Plotting logic
    plot_effect_of_the_lambda_value_on_the_inferred_ridge_regression_coefficients(design_matrix, ALPHAS, LAMBDA_VALUES, parameter_vector_coefficients)
    plot_effect_of_lambda_value_on_cross_validation_error(ALPHAS, LAMBDA_VALUES, cross_validation_error_values)

    # Now that we know the best lambda value, let's retrain with it
    retrain(design_matrix, response_vector, best_lambda_value, best_alpha)


if __name__ == "__main__":
    main()