import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#Big function to help plot fits and residuals
def plotFitAndResiduals(x, y, x_err, y_err, titleFit, titleResiduals, xlabel, ylabel, model, init_guess, equation_text, param_names, normalizedResiduals = False, reducedChiSquare = False, log_x = False, log_y = False):
    x_vals = np.array(x)
    y_vals = np.array(y)
    x_errs = np.array(x_err)
    y_errs = np.array(y_err)

    popt, pcov = curve_fit(model, x_vals, y_vals, p0 = init_guess, maxfev = 1000)

    paramErrs = np.sqrt(np.diag(pcov))
#     paramErrs = paramErrs / np.sqrt(len(x_vals))

    x_fit = np.linspace(min(x_vals), max(x_vals), 3000)
    y_fit = model(x_fit, *popt)

    y_fit_at_x = model(x_vals, *popt)
    residuals = y_vals - y_fit_at_x
    if (normalizedResiduals):
        residuals = residuals / np.std(residuals)

    plt.figure(figsize=(8.5, 6))
    plt.errorbar(x_vals, y_vals, xerr = x_errs, yerr = y_errs, label='Data', alpha = 1, fmt = 'o', ecolor = 'red', barsabove = False, markersize = 1)
    plt.plot(x_fit, y_fit, label=f'Best fit equation: {equation_text}', color='red')
    plt.title(titleFit)

    opt_params_text = ''
    for indx, name in enumerate(param_names):
        opt_params_text += f"{name} = {popt[indx]:.7f} ± {paramErrs[indx]:.7f}"
    if indx < len(param_names) - 1:
        opt_params_text += ', '

    if(reducedChiSquare):
        chi_squared = np.sum(((y_vals - y_fit_at_x) / y_errs) ** 2)
        df = len(x_vals) - len(popt)
        chi_squared_reduced = chi_squared / df
        opt_params_text += f"\nReduced Chi-squared: {chi_squared_reduced:.4f}"

#     plt.text(0.015, 0.85, opt_params_text,
#              transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left', framealpha = 0.7)

    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.show()

    plt.figure(figsize=(8.5, 3))  # Smaller height for residuals plot
    plt.errorbar(x_vals, residuals, xerr = x_errs, yerr = y_errs, label='Residuals', alpha = 1, fmt = 'o', ecolor = 'red', barsabove = False, markersize = 1)

    residsMax = np.abs(max(residuals))
    residsMin = np.abs(min(residuals))
    residsRange = max(residsMax, residsMin)

    plt.ylim(-residsRange-(residsRange/5), residsRange+(residsRange/5))

    #Add a horizontal line at y = 0 to represent a perfect fit (x-axis)
    plt.axhline(0, color='red', linestyle='--')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel + " (Residuals)")
    plt.title(titleResiduals)
    plt.legend()

    plt.show()

    return(popt, paramErrs)

#Load in ioLab data
data_filepath = '../Downloads/capstoneData'
data_list = []
#7 resistors: 1 ohm, 10 ohms, 100 ohms, 500 ohms, 1 kOhm, 5 kOhms, 10 kOhms

data_list.append(pd.read_csv(data_filepath+'/cap_1ohm2.csv'))
data_list.append(pd.read_csv(data_filepath+'/cap_10ohm2.csv'))
data_list.append(pd.read_csv(data_filepath+'/cap_100ohm2.csv'))
data_list.append(pd.read_csv(data_filepath+'/cap_500ohm2.csv'))
data_list.append(pd.read_csv(data_filepath+'/cap_1k1.csv'))
data_list.append(pd.read_csv(data_filepath+'/cap_5k2.csv'))
data_list.append(pd.read_csv(data_filepath+'/cap_10k2.csv'))

data_wire = pd.read_csv(data_filepath + "/cap_wire2.csv")

data_list[0]

"""
Visually finding start times to do the data analysis because we started recording the ioLab before
starting the experiment, so the initial data is just noise which we should eliminate before fitting!

Also, we need to "center" the data by offsetting the oscillations by gravitational acceleration!
"""

plt.plot(data_list[0]['Time (s)'][800:], data_list[0]['Ay (m/s²)'][800:])
plt.axhline(9.8, color = 'orange')
plt.show()

start_indices = [400, 1500, 900, 1100, 500, 800, 800]

#Wire data for visualization of what to expect
plt.plot(data_wire['Time (s)'], data_wire['Ay (m/s²)'])
plt.axhline(9.8, color = 'orange', label = "Gravity: 9.8 m/s^2")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration_y (m/s^2)")
plt.title("Damped Oscillations ")
plt.show()

#Fitting function (dampened harmonic oscillation acceleration model)
def fit_function(t, A, b, w, phi):
    exp_term = A*np.exp(-b*t)
    term_1 = (b**2 - w**2)*np.cos(w*t + phi)
    term_2 = 2*b*w*np.sin(w*t + phi)

    return(exp_term*(term_1 + term_2))

#243.9g +/- 0.1g oscillator mass

#1 ohm analysis
"""
x, y, x_err, y_err, titleFit, titleResiduals, xlabel, ylabel, model, init_guess, equation_text, param_names,
normalizedResiduals = False, reducedChiSquare = False
"""
input_params = [data_list[0]["Time (s)"][400:].reset_index(drop = True),
                data_list[0]["Ay (m/s²)"][400:].reset_index(drop = True)-9.8,
               [0.000001 for x in range(len(data_list[0]["Time (s)"][400:]))],
               [0.000001 for x in range(len(data_list[0]["Time (s)"][400:]))],
               "1 Ohm Damped Oscillations", "1 Ohm Damped Oscillations Residuals", "Time (s)",
               "Acceleration_y (m/s^2)", fit_function, [2,0.1,1,0],
                "a_y(t) = Ae^(-bt)[(b^2 - w^2)cos(wt + phi) + 2bwsin(wt + phi)]", ['A', 'b', 'w', 'phi'],
               False, False]
ohm1 = plotFitAndResiduals(*input_params)

#10 ohm analysis
"""
x, y, x_err, y_err, titleFit, titleResiduals, xlabel, ylabel, model, init_guess, equation_text, param_names,
normalizedResiduals = False, reducedChiSquare = False
"""
input_params = [data_list[1]["Time (s)"][1500:].reset_index(drop = True),
                data_list[1]["Ay (m/s²)"][1500:].reset_index(drop = True)-9.8,
               [0.000001 for x in range(len(data_list[1]["Time (s)"][1500:]))],
               [0.000001 for x in range(len(data_list[1]["Time (s)"][1500:]))],
               "10 Ohms Damped Oscillations", "10 Ohms Damped Oscillations Residuals", "Time (s)",
               "Acceleration_y (m/s^2)", fit_function, [2,0.1,1,0],
                "a_y(t) = Ae^(-bt)[(b^2 - w^2)cos(wt + phi) + 2bwsin(wt + phi)]", ['A', 'b', 'w', 'phi'],
               False, False]
ohm10 = plotFitAndResiduals(*input_params)

#100 ohm analysis
"""
x, y, x_err, y_err, titleFit, titleResiduals, xlabel, ylabel, model, init_guess, equation_text, param_names,
normalizedResiduals = False, reducedChiSquare = False
"""
input_params = [data_list[2]["Time (s)"][900:].reset_index(drop = True),
                data_list[2]["Ay (m/s²)"][900:].reset_index(drop = True)-9.8,
               [0.000001 for x in range(len(data_list[2]["Time (s)"][900:]))],
               [0.000001 for x in range(len(data_list[2]["Time (s)"][900:]))],
               "100 Ohms Damped Oscillations", "100 Ohms Damped Oscillations Residuals", "Time (s)",
               "Acceleration_y (m/s^2)", fit_function, [2,0.1,1,0],
                "a_y(t) = Ae^(-bt)[(b^2 - w^2)cos(wt + phi) + 2bwsin(wt + phi)]", ['A', 'b', 'w', 'phi'],
               False, False]
ohm100 = plotFitAndResiduals(*input_params)

#500 ohm analysis
"""
x, y, x_err, y_err, titleFit, titleResiduals, xlabel, ylabel, model, init_guess, equation_text, param_names,
normalizedResiduals = False, reducedChiSquare = False
"""
input_params = [data_list[3]["Time (s)"][1100:].reset_index(drop = True),
                data_list[3]["Ay (m/s²)"][1100:].reset_index(drop = True)-9.8,
               [0.000001 for x in range(len(data_list[3]["Time (s)"][1100:]))],
               [0.000001 for x in range(len(data_list[3]["Time (s)"][1100:]))],
               "500 Ohms Damped Oscillations", "500 Ohms Damped Oscillations Residuals", "Time (s)",
               "Acceleration_y (m/s^2)", fit_function, [2,0.1,1,0],
                "a_y(t) = Ae^(-bt)[(b^2 - w^2)cos(wt + phi) + 2bwsin(wt + phi)]", ['A', 'b', 'w', 'phi'],
               False, False]
ohm500 = plotFitAndResiduals(*input_params)

#1 K-ohm analysis
"""
x, y, x_err, y_err, titleFit, titleResiduals, xlabel, ylabel, model, init_guess, equation_text, param_names,
normalizedResiduals = False, reducedChiSquare = False
"""
input_params = [data_list[4]["Time (s)"][start_indices[4]:].reset_index(drop = True),
                data_list[4]["Ay (m/s²)"][start_indices[4]:].reset_index(drop = True)-9.8,
               [0.000001 for x in range(len(data_list[4]["Time (s)"][start_indices[4]:]))],
               [0.000001 for x in range(len(data_list[4]["Time (s)"][start_indices[4]:]))],
               "1 Kilo-ohm Damped Oscillations", "1 Kilo-ohm Damped Oscillations Residuals", "Time (s)",
               "Acceleration_y (m/s^2)", fit_function, [2,0.1,1,0],
                "a_y(t) = Ae^(-bt)[(b^2 - w^2)cos(wt + phi) + 2bwsin(wt + phi)]", ['A', 'b', 'w', 'phi'],
               False, False]
ohm1k = plotFitAndResiduals(*input_params)

#5 K-ohm analysis
"""
x, y, x_err, y_err, titleFit, titleResiduals, xlabel, ylabel, model, init_guess, equation_text, param_names,
normalizedResiduals = False, reducedChiSquare = False
"""
input_params = [data_list[5]["Time (s)"][start_indices[5]:].reset_index(drop = True),
                data_list[5]["Ay (m/s²)"][start_indices[5]:].reset_index(drop = True)-9.8,
               [0.000001 for x in range(len(data_list[5]["Time (s)"][start_indices[5]:]))],
               [0.000001 for x in range(len(data_list[5]["Time (s)"][start_indices[5]:]))],
               "5 Kilo-ohms Damped Oscillations", "5 Kilo-ohms Damped Oscillations Residuals", "Time (s)",
               "Acceleration_y (m/s^2)", fit_function, [2,0.1,1,0],
                "a_y(t) = Ae^(-bt)[(b^2 - w^2)cos(wt + phi) + 2bwsin(wt + phi)]", ['A', 'b', 'w', 'phi'],
               False, False]
ohm5k = plotFitAndResiduals(*input_params)

#10 K-ohm analysis
"""
x, y, x_err, y_err, titleFit, titleResiduals, xlabel, ylabel, model, init_guess, equation_text, param_names,
normalizedResiduals = False, reducedChiSquare = False
"""
input_params = [data_list[6]["Time (s)"][start_indices[6]:].reset_index(drop = True),
                data_list[6]["Ay (m/s²)"][start_indices[6]:].reset_index(drop = True)-9.8,
               [0.000001 for x in range(len(data_list[6]["Time (s)"][start_indices[6]:]))],
               [0.000001 for x in range(len(data_list[6]["Time (s)"][start_indices[6]:]))],
               "10 Kilo-ohms Damped Oscillations", "10 Kilo-ohms Damped Oscillations Residuals", "Time (s)",
               "Acceleration_y (m/s^2)", fit_function, [2,0.1,1,0],
                "a_y(t) = Ae^(-bt)[(b^2 - w^2)cos(wt + phi) + 2bwsin(wt + phi)]", ['A', 'b', 'w', 'phi'],
               False, False]
ohm10k = plotFitAndResiduals(*input_params)

fit_data = [ohm1, ohm10, ohm100, ohm500, ohm1k, ohm5k, ohm10k]

ohm1

def error_prop_gamma(beta, beta_err):
    return(2*np.sqrt((0.2439**2)*(beta_err**2) + (beta**2)*(0.0001**2)))

y_data = []
y_err = []
for i in fit_data:
    y_data.append(i[0][1])
    y_err.append(i[1][1])
y_data = np.array(y_data)
y_err = np.array(y_err)
y_err = error_prop_gamma(y_data, y_err) #error-propagation to account for damping coefficient error
y_data = y_data * 2 * 0.2439 #Account for mass to get damping factor
x_data = [1.1, 12.1, 100.3, 477.3, 1003, 5075, 9920]
x_err = [0.1, 0.1, 0.1, 0.1, 1, 1, 10]

plt.scatter(x_data, y_data)
plt.title("Damping vs. Resistance")
plt.xlabel("Resistance (Ohms)")
plt.ylabel("Damping Factor (kg/s)")
plt.show()

def inverse_function(x, A):
    return(A/x)

def power_law(x, A, p):
    return(A/(x**p))

input_params = [x_data,
                y_data,
               x_err,
               y_err,
               "Damping vs. Resistance (Power Law Fit)", "Damping vs. Resistance Residuals (Power Law Fit)", "Resistance (Ohms)",
               "Damping Factor (kg/s)", power_law, [0.1, 0.1],
                "gamma = A * 1/R^p", ['A', 'p'],
               False, False, False, False]

power_law_fit = plotFitAndResiduals(*input_params)

input_params = [x_data,
                y_data,
               x_err,
               y_err,
               "Damping vs. Resistance (Inverse Fit)", "Damping vs. Resistance Residuals (Inverse Fit)", "Resistance (Ohms)",
               "Damping Factor (kg/s)", inverse_function, [0.1],
                "gamma = A * 1/R", ['A'],
               False, False, False, False]

inverse_fit = plotFitAndResiduals(*input_params)

def printFittedParameters(results):
    vals = results[0]
    errs = results[1]
    print(f"\nA = {vals[0]} +/- {errs[0]}")
    print(f"b = {vals[1]} +/- {errs[1]}")
    print(f"w = {vals[2]} +/- {errs[2]}")
    print(f"phi = {vals[3]} +/- {errs[3]}")

for x in fit_data:
    printFittedParameters(x)

print(x_data, "\n", x_err)
print(y_data, "\n", y_err)

print(inverse_fit)

print(power_law_fit)

-0.15375 + 1