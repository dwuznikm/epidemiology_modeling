import streamlit as st
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import cv2
import csv


def sir_model_with_vaccination(y, t, params):
    S, E, I, R, V, V_failed, D = y
    beta = params["beta"]
    sigma = params["sigma"]
    gamma = params["gamma"]
    alpha = params["alpha"]
    v_rate = params["v_rate"]
    v_success = params["v_success"]
    alpha_v = params["alpha_v"]
    alpha_v_failed = params["alpha_v_failed"]
    delta = params["delta"]

    dSdt = (
        -beta * S * I - v_rate * S + alpha_v * V + alpha * R + alpha_v_failed * V_failed
    )
    dEdt = beta * S * I - sigma * E + V_failed * beta * I
    dIdt = sigma * E - gamma * I - delta * I
    dRdt = gamma * I - alpha * R
    dDdt = delta * I
    dVdt = v_rate * S * v_success - alpha_v * V
    dV_failedt = (
        v_rate * S * (1 - v_success) - V_failed * beta * I - alpha_v_failed * V_failed
    )
    return [dSdt, dEdt, dIdt, dRdt, dVdt, dV_failedt, dDdt]


def plot_sir_model_simulation(result, days, title_addon="", highlight=None):
    t = np.linspace(0, days, days)
    result_percentage = (result / population_size) * 100
    plt.figure(figsize=(10, 6))
    labels = [
        "Susceptible",
        "Exposed",
        "Infected",
        "Recovered",
        "Vaccinated",
        "Failed vaccination",
        "Dead",
    ]
    for i in range(result_percentage.shape[1]):
        if highlight == labels[i]:
            plt.plot(t, result_percentage[:, i], label=labels[i], lw=3)
        else:
            plt.plot(t, result_percentage[:, i], label=labels[i])

    plt.xlabel("Time (days)")
    plt.ylabel("Percentage of Population")
    plt.title(f"SIR Model{title_addon}")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


def generate_sir_model_data(
    population_size, params, days, y0=None, seasonal_amplitude=False, noise_std=False
):
    t = np.linspace(0, days, days)
    if y0 is None:
        y0 = [population_size - 1, 0, 1, 0, 0, 0, 0]

    result = odeint(sir_model_with_vaccination, y0, t, args=(params,))
    strength_factor = max(0.1, 0.1 * (population_size / 40))
    if seasonal_amplitude:
        seasonal_variation = strength_factor * np.sin(2 * np.pi * t / 365)
        result += seasonal_variation[:, np.newaxis]

    if noise_std:
        noise = np.random.normal(scale=strength_factor, size=result.shape)
        result += noise

    result[result < 0] = 0
    df = pd.DataFrame(
        result,
        columns=[
            "Susceptible",
            "Exposed",
            "Infected",
            "Recovered",
            "Vaccinated",
            "Failed vaccination",
            "Dead",
        ],
    )

    return df, result


def plot_from_params(population_size, params, days, title_addon=""):
    df, result = generate_sir_model_data(population_size, params[0], days)
    plot_sir_model_simulation(result, days, title_addon)
    st.write("Summed Error:", params[2])
    st.write("Parameters:", params[0])


def bilateral_filter(df, kernel_size=23, sigma_color=777777, sigma_space=70):
    filtered_df = pd.DataFrame(index=df.index)  # Set the index explicitly
    for col in df.columns:
        filtered_col = cv2.bilateralFilter(
            df[col].values.astype(np.float32), kernel_size, sigma_color, sigma_space
        )
        filtered_df[col] = filtered_col
    return filtered_df


def mse_loss(params, t, data, y0):
    params_dict = {
        "beta": params[0],
        "sigma": params[1],
        "gamma": params[2],
        "alpha": params[3],
        "v_rate": params[4],
        "v_success": params[5],
        "alpha_v": params[6],
        "alpha_v_failed": params[7],
        "delta": params[8],
    }
    result = odeint(sir_model_with_vaccination, y0, t, args=(params_dict,))
    loss = np.mean(
        (result[:, 0] - data["Susceptible"]) ** 2
        + (result[:, 1] - data["Exposed"]) ** 2
        + (result[:, 2] - data["Infected"]) ** 2
        + (result[:, 3] - data["Recovered"]) ** 2
        + (result[:, 4] - data["Vaccinated"]) ** 2
        + (result[:, 5] - data["Failed vaccination"]) ** 2
        + (result[:, 6] - data["Dead"]) ** 2
    )
    return loss


def optimize_params(data, population_size, initial_params, og_params):
    days = len(data)
    t = np.linspace(0, days, days)

    data = bilateral_filter(data)
    y0 = [population_size - 1, 0, 1, 0, 0, 0, 0]

    bounds = [
        (0, 0.1),  # beta
        (0, 0.4),  # sigma
        (0, 0.4),  # gamma
        (0, 0.4),  # alpha
        (0, 0.1),  # v_rate
        (0, 1),  # v_success
        (0, 0.4),  # alpha_v
        (0, 0.4),  # alpha_v_failed
        (0, 0.2),  # delta
    ]
    lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    upper_bounds = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    result_ls = least_squares(
        mse_loss,
        initial_params,
        args=(t, data, y0),
        bounds=(lower_bounds, upper_bounds),
    )
    result_min = minimize(
        mse_loss, initial_params, args=(t, data, y0), bounds=bounds, method="L-BFGS-B"
    )
    result_diff = differential_evolution(
        mse_loss, bounds, args=(t, data, y0), maxiter=500, tol=0.01, workers=-1
    )
    params_ls = create_params_dict(result_ls.x)
    params_min = create_params_dict(result_min.x)
    params_diff = create_params_dict(result_diff.x)
    error_min = calculate_summed_error(og_params, params_min)
    error_diff = calculate_summed_error(og_params, params_diff)
    error_ls = calculate_summed_error(og_params, params_ls)
    errors = [
        (params_min, "minimize", error_min),
        (params_diff, "differential_evolution", error_diff),
        (params_ls, "least_squares", error_ls),
    ]

    return errors


def create_params_dict(optimized_params):
    params_dict = {
        "beta": optimized_params[0],
        "sigma": optimized_params[1],
        "gamma": optimized_params[2],
        "alpha": optimized_params[3],
        "v_rate": optimized_params[4],
        "v_success": optimized_params[5],
        "alpha_v": optimized_params[6],
        "alpha_v_failed": optimized_params[7],
        "delta": optimized_params[8],
    }
    return params_dict


def calculate_summed_error(og_params, optimized_params):
    summed_error = sum(
        abs(float(optimized_params[key]) - float(og_params[key])) for key in og_params
    )
    return summed_error


st.title("SIR Model")

scenario_option = st.selectbox(
    "Choose a scenario:",
    (
        "Hospitalizations and social restrictions",
        "Seasonality",
        "Vaccination rate vs vaccination success",
        "Advanced",
        "Prediction",
    ),
)


if scenario_option == "Hospitalizations and social restrictions":
    population_size = 10000
    prob_of_infecting = 0.02
    avg_no_contacts_per_individual = 20
    beta = prob_of_infecting * avg_no_contacts_per_individual / population_size
    params = {
        "beta": beta,
        "sigma": 0.15,
        "gamma": 0.05,
        "alpha": 0.01,
        "v_rate": 0.001,
        "v_success": 0.70,
        "alpha_v": 0.005,
        "alpha_v_failed": 0.001,
        "delta": 0.03,
    }
    seasonal_amplitude = False
    noise_std = False
    days = 1500
    df_no_restrictions, result1 = generate_sir_model_data(
        population_size,
        params,
        days,
        seasonal_amplitude=seasonal_amplitude,
        noise_std=noise_std,
    )
    with open("params_no_restrictions.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])
    params["beta"] = 0.02 * 12 / population_size
    df_movement_restrictions, result2 = generate_sir_model_data(
        population_size,
        params,
        days,
        seasonal_amplitude=seasonal_amplitude,
        noise_std=noise_std,
    )
    with open("params_movement_restrictions.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])
    params["beta"] = 0.01 * 12 / population_size
    df_movement_restrictions_masks, result3 = generate_sir_model_data(
        population_size,
        params,
        days,
        seasonal_amplitude=seasonal_amplitude,
        noise_std=noise_std,
    )
    with open("params_movement_restrictions_masks.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])

    option = st.radio("Choose an option:", ("Simulation", "Data Generation"))
    if option == "Simulation":
        if st.button("Run Simulation"):
            plot_sir_model_simulation(result1, days, " no social restrictions", "Infected")
            plot_sir_model_simulation(result2, days, " movement restrictions", "Infected")
            plot_sir_model_simulation(result3, days, " movement restrictions and masks", "Infected")
    if option == "Data Generation":
        if st.button("Generate Data"):
            df_no_restrictions.to_csv("data_no_restrictions.csv", index=False)
            df_movement_restrictions.to_csv("data_movement_restrictions.csv", index=False)
            df_movement_restrictions_masks.to_csv("data_movement_restrictions_masks.csv", index=False)

elif scenario_option == "Seasonality":
    population_size = 10000
    prob_of_infecting = 0.01
    avg_no_contacts_per_individual = 12
    beta = prob_of_infecting * avg_no_contacts_per_individual / population_size
    params = {
        "beta": beta,
        "sigma": 0.15,
        "gamma": 0.05,
        "alpha": 0.01,
        "v_rate": 0.001,
        "v_success": 0.70,
        "alpha_v": 0.005,
        "alpha_v_failed": 0.001,
        "delta": 0.03,
    }
    seasonal_amplitude = False
    noise_std = False
    days = 3000
    parameter_name = st.selectbox("Choose parameter:", list(params.keys()))
    num_changes_per_year = st.select_slider(
        "Number of changes per year:", options=[2, 4]
    )
    values = []
    for i in range(num_changes_per_year):
        value = st.number_input(
            f"Value for year part {i+1}:",
            value=params[parameter_name],
            format="%.3f",
            step=0.001,
        )
        values.append(value)
    if num_changes_per_year == 2:
        days_per_cycle = 183
    else:
        days_per_cycle = 91
    if st.button("Run Simulation"):
        total_result = None
        counter = 0
        for i in range(math.ceil(days / days_per_cycle)):
            params[parameter_name] = values[counter]
            counter += 1
            if counter == num_changes_per_year:
                counter = 0
            if total_result is None:
                df, result = generate_sir_model_data(
                    population_size,
                    params,
                    days_per_cycle,
                    seasonal_amplitude=seasonal_amplitude,
                    noise_std=noise_std,
                )
                total_result = result
            else:
                df, result = generate_sir_model_data(
                    population_size,
                    params,
                    days_per_cycle,
                    y0=total_result[-1],
                    seasonal_amplitude=seasonal_amplitude,
                    noise_std=noise_std,
                )
                total_result = np.concatenate((total_result, result))
        plot_sir_model_simulation(
            total_result, len(total_result), highlight=parameter_name
        )

elif scenario_option == "Vaccination rate vs vaccination success":
    population_size = 10000
    prob_of_infecting = 0.01
    avg_no_contacts_per_individual = 13
    beta = prob_of_infecting * avg_no_contacts_per_individual / population_size
    params = {
        "beta": beta,
        "sigma": 0.14,
        "gamma": 0.05,
        "alpha": 0.02,
        "v_rate": 0.0,
        "v_success": 0.0,
        "alpha_v": 0.005,
        "alpha_v_failed": 0.001,
        "delta": 0.03,
    }
    seasonal_amplitude = False
    noise_std = False
    days = 1500
    df_no_vacc, result1 = generate_sir_model_data(
        population_size,
        params,
        days,
        seasonal_amplitude=seasonal_amplitude,
        noise_std=noise_std,
    )
    with open("params_no_vacc.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])
    params["v_rate"] = 0.004
    params["v_success"] = 0.3
    with open("params_high_v_rate_low_v_success.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])
    df_high_v_rate_low_v_success, result2 = generate_sir_model_data(
        population_size,
        params,
        days,
        seasonal_amplitude=seasonal_amplitude,
        noise_std=noise_std,
    )
    params["v_rate"] = 0.001
    params["v_success"] = 0.9
    with open("params_low_v_rate_high_v_success.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])
    df_low_v_rate_high_v_success, result3 = generate_sir_model_data(
        population_size,
        params,
        days,
        seasonal_amplitude=seasonal_amplitude,
        noise_std=noise_std,
    )
    params["v_rate"] = 0.004
    params["v_success"] = 0.9
    with open("params_high_v_rate_high_v_success.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])
    df_high_v_rate_high_v_success, result4 = generate_sir_model_data(
        population_size,
        params,
        days,
        seasonal_amplitude=seasonal_amplitude,
        noise_std=noise_std,
    )

    option = st.radio("Choose an option:", ("Simulation", "Data Generation"))
    if option == "Simulation":
        if st.button("Run Simulation"):
            plot_sir_model_simulation(result1, days, " no vaccinations", "Dead")
            plot_sir_model_simulation(
                result2, days, " high v_rate low v_success", "Dead"
            )
            plot_sir_model_simulation(
                result3, days, " low v_rate high v_success", "Dead"
            )
            plot_sir_model_simulation(
                result4, days, " high v_rate high v_success", "Dead"
            )
    if option == "Data Generation":
        if st.button("Generate Data"):
            df_no_vacc.to_csv("data_no_vacc.csv", index=False)
            df_high_v_rate_low_v_success.to_csv(
                "data_high_v_rate_low_v_success.csv", index=False
            )
            df_low_v_rate_high_v_success.to_csv(
                "data_low_v_rate_high_v_success.csv", index=False
            )
            df_high_v_rate_high_v_success.to_csv(
                "data_high_v_rate_high_v_success.csv", index=False
            )
elif scenario_option == "Advanced":
    option = st.radio("Choose an option:", ("Simulation", "Data Generation"))
    population_size = st.number_input(
        "Population Size", value=int(3.6 * 10**7), step=1000000
    )
    prob_of_infecting = st.number_input(
        "Probability of Infecting", value=1 / 100, format="%.3f", step=0.001
    )
    avg_no_contacts_per_individual = st.number_input(
        "Average Number of Contacts per Individual", value=12
    )
    beta = prob_of_infecting * avg_no_contacts_per_individual / population_size
    params = {
        "beta": beta,
        "sigma": st.number_input(
            "Infection Rate (sigma)",
            min_value=0.0,
            max_value=1.0,
            value=0.14,
            step=0.01,
        ),
        "gamma": st.number_input(
            "Recovery Rate (gamma)",
            min_value=0.0,
            max_value=1.0,
            value=1 / 21,
            step=0.01,
        ),
        "alpha": st.number_input(
            'Rate at which individuals leave the "Recovered" compartment (alpha)',
            min_value=0.0,
            max_value=0.5,
            value=0.0055,
            format="%.3f",
            step=0.001,
        ),
        "alpha_v": st.number_input(
            'Rate at which individuals leave the "Vaccinated" compartment (alpha_v)',
            min_value=0.0,
            max_value=0.5,
            value=0.005,
            format="%.3f",
            step=0.001,
        ),
        "alpha_v_failed": st.number_input(
            'Rate at which individuals leave the "Failed vaccination" compartment - enables them to revaccinate (alpha_v_failed)',
            min_value=0.0,
            max_value=0.5,
            value=0.002,
            format="%.3f",
            step=0.001,
        ),
        "v_rate": st.number_input(
            "Vaccination Rate (v_rate)",
            min_value=0.0,
            max_value=0.1,
            value=0.001,
            format="%.3f",
            step=0.001,
        ),
        "v_success": st.number_input(
            "Vaccination Success Rate (v_success)",
            min_value=0.0,
            max_value=1.0,
            value=0.72,
            step=0.01,
        ),
        "delta": st.number_input(
            "Mortality Rate (delta)",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            format="%.3f",
            step=0.001,
        ),
    }
    seasonal_amplitude = st.checkbox("Include Seasonal Amplitude")
    noise_std = st.checkbox("Include Noise Standard Deviation")
    days = st.number_input("Number of Days", value=3000, step=10)

    if option == "Simulation":
        if st.button("Run Simulation"):
            df, result = generate_sir_model_data(
                population_size,
                params,
                days,
                seasonal_amplitude=seasonal_amplitude,
                noise_std=noise_std,
            )
            plot_sir_model_simulation(result, days)

    else:
        if st.button("Generate Data"):
            df, result = generate_sir_model_data(
                population_size,
                params,
                days,
                seasonal_amplitude=seasonal_amplitude,
                noise_std=noise_std,
            )
            st.write(df)
            df.to_csv("data.csv", index=False)
            with open("params.csv", "w") as csv_file:
                writer = csv.writer(csv_file)
                for key, value in params.items():
                    writer.writerow([key, value])
elif scenario_option == "Prediction":
    csv_files = [f for f in os.listdir() if f.endswith(".csv") and "params" not in f]
    selected_file = st.selectbox("Select a CSV file:", csv_files)

    if selected_file:
        data = pd.read_csv(selected_file)
        st.write(data)
        population_size = data["Susceptible"][0]
        initial_params = [
            st.number_input("Initial Beta", value=0.000001, format="%.6f"),
            st.number_input("Initial Sigma", value=0.15, format="%.2f"),
            st.number_input("Initial Gamma", value=0.05, format="%.2f"),
            st.number_input("Initial Alpha", value=0.01, format="%.2f"),
            st.number_input("Initial Vaccination Rate", value=0.001, format="%.3f"),
            st.number_input(
                "Initial Vaccination Success Rate", value=0.70, format="%.2f"
            ),
            st.number_input("Initial Alpha V", value=0.005, format="%.3f"),
            st.number_input("Initial Alpha V Failed", value=0.001, format="%.3f"),
            st.number_input("Initial Delta", value=0.03, format="%.2f"),
        ]

        if st.button("Predict Parameters"):
            params_file = selected_file.replace("data", "params")
            with open(params_file) as csv_file:
                reader = csv.reader(csv_file)
                og_params = dict(reader)

            all_params = optimize_params(
                data, population_size, initial_params, og_params
            )

            print(all_params)
            best_params = min(all_params, key=lambda x: x[2])
            st.write("Best method:", best_params[1])
            days = len(data)
            t = np.linspace(0, days, days)
            y0 = [population_size - 1, 0, 1, 0, 0, 0, 0]
            plot_from_params(
                population_size, all_params[2], days, " predicted using least_squares"
            )
            result_diff = odeint(
                sir_model_with_vaccination, y0, t, args=(all_params[1][0],)
            )
            plot_from_params(
                population_size, all_params[0], days, " predicted using minimize"
            )
            plot_from_params(
                population_size,
                all_params[1],
                days,
                " predicted using differential_evolution",
            )
            plot_sir_model_simulation(data.values, days, " Actual Data")
            st.write("Original Parameters:", og_params)
