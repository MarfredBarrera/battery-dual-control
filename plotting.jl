using JLD2
using Statistics
using LaTeXStrings
@load "simulation_results.jld2" x_rec u_rec cov_rec x_true_rec cost_rec est_err_rec x_rec_mpc u_rec_mpc cov_rec_mpc x_true_rec_mpc cost_rec_mpc est_err_rec_mpc

### Grab required parameters from main.jl
num_simulations = length(x_rec)
T = length(x_rec[1])
n = length(x_rec[1][1])
set_point = 0.8*ones(n)
running_cost = (x, cov, u) -> (x-set_point)' * Q * (x-set_point)  + tr(Q*cov) + u' * R * u

# Calculate means
mean_est_err_rec = sum(est_err_rec)/num_simulations
mean_est_err_rec_mpc = sum(est_err_rec_mpc)/num_simulations
mean_cost_rec = sum(cost_rec)/num_simulations
mean_cost_rec_mpc = sum(cost_rec_mpc)/num_simulations


### Use to plot histograms of cost and estimation error to compare both methods
function histograms()
    # Create a 2x2 layout for the subplots
    plot_layout = @layout [a b]


    # Define consistent bin edges for estimation error histograms
    num_bins = 21
    min_estimation = minimum([minimum(est_err_rec), minimum(est_err_rec_mpc)])
    max_estimation = maximum([maximum(est_err_rec), maximum(est_err_rec_mpc)])
    bins_estimation = range(min_estimation, (max_estimation + (max_estimation - min_estimation) / (num_bins - 1)), length=num_bins)

    # Define consistent bin edges for cost histograms
    min_cost = minimum([minimum(cost_rec), minimum(cost_rec_mpc)])
    max_cost = maximum([maximum(cost_rec), maximum(cost_rec_mpc)])
    bins_cost = range(min_cost, max_cost + (max_cost - min_cost) / (num_bins - 1), length=num_bins)  # Add one extra bin to the right

    # Individual histograms with vertical dashed lines for means
    p1 = histogram(est_err_rec, 
        bins=bins_estimation, 
        label="Stochastic Optimal Control", 
        color=:blue, 
        xlabel="Estimation Error", 
        ylabel="Frequency",
        legend=:topright)
    vline!(p1, [mean_est_err_rec], label="Mean", color=:red, linestyle=:dash, linewidth=1.5)
    ylims!(0.0,60)

    p2 = histogram(est_err_rec_mpc, 
        bins=bins_estimation, 
        label="Model Predictive Control", 
        color=:green, 
        xlabel="Estimation Error", 
        ylabel="Frequency",
        legend=:topright)
    vline!(p2, [mean_est_err_rec_mpc], label="Mean", color=:red, linestyle=:dash, linewidth=1.5)
    ylims!(0.0,60)


    p3 = histogram(cost_rec, 
        bins=bins_cost, 
        label="Stochastic Optimal Control", 
        color=:blue, 
        xlabel="Cost", 
        ylabel="Frequency")
    vline!(p3, [mean_cost_rec], label="Mean", color=:red, linestyle=:dash,linewidth=1.5)
    ylims!(0.0,60)

    p4 = histogram(cost_rec_mpc, 
        bins=bins_cost, 
        label="Model Predictive Control", 
        color=:green, 
        xlabel="Cost", 
        ylabel="Frequency")
    vline!(p4, [mean_cost_rec_mpc], label="Mean", color=:red, linestyle=:dash,linewidth=1.5)
    ylims!(0.0,60)

    plot(p1, p3,
        layout=plot_layout, 
        legend=:topright,
        legendfontsize=7)
    savefig("./SOC_histograms.png")

    plot(p2,p4,
        layout=plot_layout, 
        legend=:topright,
        legendfontsize=7)

    savefig("./MPC_histograms.png")

end



function simulation_averaged_cost()
    # Initialize a vector to store the average running cost for each time step
    avg_running_cost = zeros(T)
    avg_running_cost_mpc = zeros(T)

    # Loop over all simulations and compute the running cost for each time step
    for i in 1:num_simulations
        for t in 1:T
            # Compute the running cost for the current simulation and time step
            cost = running_cost(x_true_rec[i][t], 0, u_rec[i][t]) 
            cost_mpc = running_cost(x_true_rec_mpc[i][t], 0, u_rec_mpc[i][t])
            # Accumulate the running cost
            avg_running_cost[t] += cost
            avg_running_cost_mpc[t] += cost_mpc
        end
    end

    # Divide by the number of simulations to get the average
    avg_running_cost ./= num_simulations
    avg_running_cost_mpc ./= num_simulations

    # Plot the average running cost across time
    plot(1:T, avg_running_cost, 
    xlabel="Time Step", 
    ylabel="Average Running Cost", 
    label="Stochastic Optimal Control", color=:blue)
    plot!(1:T, avg_running_cost_mpc, label="Model Predictive Control", color=:green)

    ylims!(0.0,0.3)
    savefig("./average_running_cost.png")
end


function simulation_averaged_estimation_err()
    # Initialize a vector to store the average running cost for each time step
    avg_estimation_err = zeros(T)
    avg_estimation_err_mpc = zeros(T)

    # Loop over all simulations and compute the running cost for each time step
    for i in 1:num_simulations
        for t in 1:T
            # Accumulate the running cost
            avg_estimation_err[t] += norm(x_rec[i][t] - x_true_rec[i][t])
            avg_estimation_err_mpc[t] += norm(x_rec_mpc[i][t] - x_true_rec_mpc[i][t])
        end
    end
    # print(avg_estimation_err)

    # # Divide by the number of simulations to get the average
    avg_estimation_err ./= num_simulations
    avg_estimation_err_mpc ./= num_simulations 

    # Plot the average running cost across time
    plot(1:T, avg_estimation_err, 
    xlabel="Time Step", 
    ylabel="Estimation Error", 
    label="Stochastic Optimal Control", color=:blue)
    plot!(1:T, avg_estimation_err_mpc, 
    label="Model Predictive Control", 
    color=:green)

    savefig("./estimation_error.png")

end

# simulation_averaged_cost()

function simulation_averaged_covariance()
    # Initialize a 3D array to store the cumulative covariance matrices for each time step
    avg_covariance = zeros(size(cov_rec[1][1])..., T)  # Shape: (n, n, T)
    avg_covariance_mpc = zeros(size(cov_rec_mpc[1][1])..., T)  # Shape: (n, n, T)

    # Loop over all simulations and accumulate the covariance matrices
    for i in 1:num_simulations
        for t in 1:T
            avg_covariance[:, :, t] += cov_rec[i][t]
            avg_covariance_mpc[:, :, t] += cov_rec_mpc[i][t]
        end
    end

    # Divide by the number of simulations to get the average covariance matrix for each time step
    avg_covariance ./= num_simulations
    avg_covariance_mpc ./= num_simulations

    var1 = avg_covariance[1, 1, :]
    var1_mpc = avg_covariance_mpc[1, 1, :]

    # Plot the variance of the first state over time with LaTeX labels
    plot(1:T, var1, 
    label="Variance of x₁ (Stochastic Optimal Control)", 
    color=:blue, 
    xlabel="Time Step", 
    ylabel="Variance")

    plot!(1:T, var1_mpc, 
    label="Variance of x₁ (Model Predictive Control)", 
    color=:green,
    xlabel="Time Step",
    ylabel="Variance",)

    ylims!(0.0,0.05)

    savefig("./covariance.png")
end

function plot_OCV_SOC()
    # Define the SOC range
    SOC_range = 0:0.01:1  # SOC values from 0 to 1 with a step of 0.01

    # Compute OCV values for each SOC in the range
    OCV_values = [measurement_dynamics([SOC, SOC, SOC, SOC, SOC, SOC]) for SOC in SOC_range]
    OCV_values = reduce(hcat, OCV_values)  # Combine into a matrix for easier plotting

    # Plot each OCV-SOC curve
    plot(
        SOC_range, OCV_values[1, :], 
        label="Lithium-Ion, Wang", 
        linewidth=2, 
        linestyle=:solid, 
        color=:blue
    )
    plot!(
        SOC_range, OCV_values[2, :], 
        label="Lithium-Ion, Zhang, Exponential", 
        linewidth=2, 
        linestyle=:dash, 
        color=:red
    )
    plot!(
        SOC_range, OCV_values[3, :], 
        label="Lithium-Ion, Zhang, Sum of Sines",
        linewidth=2, 
        linestyle=:dot, 
        color=:green
    )
    plot!(
        SOC_range, OCV_values[4, :], 
        label="Lithium-Ion, Zhao, Battery 1", 
        linewidth=2, 
        linestyle=:dashdot, 
        color=:purple
    )
    plot!(
        SOC_range, OCV_values[5, :], 
        label="Lithium-Ion, Zhao, Battery 2", 
        linewidth=2, 
        linestyle=:solid, 
        color=:orange
    )
    plot!(
        SOC_range, OCV_values[6, :], 
        label="Lithium Titanate Oxide, Stroe", 
        linewidth=2, 
        linestyle=:solid, 
        color=:cyan
    )

    # plot!(
    #     SOC_range, OCV_values[7, :], 
    #     label="LTO, tanh", 
    #     linewidth=2, 
    #     linestyle=:solid, 
    #     color=:magenta
    # )
    # plot!(
    #     SOC_range, OCV_values[8, :], 
    #     label="LCO, sin", 
    #     linewidth=2, 
    #     linestyle=:solid, 
    #     color=:brown
    # )

    # # Add plot aesthetics
    # plot!(
    #     xlabel="State of Charge (SOC)", 
    #     ylabel="Open Circuit Voltage (OCV)", 
    #     title="OCV-SOC Curves for Observation Models", 
    #     legend=:topright, 
    #     grid=:on, 
    #     framestyle=:box, 
    #     size=(800, 600), 
    #     tickfontsize=10, 
    #     guidefontsize=12, 
    #     titlefontsize=14
    # )

xlabel!("State of Charge (SOC)")
ylabel!("Open Circuit Voltage (OCV)")
# Save the plot
    savefig("./OCV.png")
end


histograms()
simulation_averaged_cost()
simulation_averaged_covariance()
simulation_averaged_estimation_err()
plot_OCV_SOC()

