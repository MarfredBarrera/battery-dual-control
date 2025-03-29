using JLD2

# load saved simulation data
@load "simulation_results.jld2" x_rec u_rec cov_rec x_true_rec cost_rec est_err_rec x_rec_mpc u_rec_mpc cov_rec_mpc x_true_rec_mpc cost_rec_mpc est_err_rec_mpc


# Calculate means
mean_est_err_rec = sum(est_err_rec)/num_simulations
mean_est_err_rec_mpc = sum(est_err_rec_mpc)/num_simulations
mean_cost_rec = sum(cost_rec)/num_simulations
mean_cost_rec_mpc = sum(cost_rec_mpc)/num_simulations


x1 = [x_rec[1][t][1] for t in 1:T]
x2 = [x_rec[1][t][2] for t in 1:T]

x1_mpc = [x_rec_mpc[1][t][1] for t in 1:T]
x2_mpc = [x_rec_mpc[1][t][2] for t in 1:T]

running_cost = [CS.running_cost(x_true_rec[10][k], 0, u_rec[10][k]) for k in 1:T]
running_cost_mpc = [CS.running_cost(x_true_rec_mpc[10][k], 0, u_rec_mpc[10][k]) for k in 1:T]

variance = [[cov_rec[10][k][1,1]; cov_rec[10][k][2,2]] for k in 1:T]
variance_mpc = [[cov_rec_mpc[10][k][1,1]; cov_rec_mpc[10][k][2,2]] for k in 1:T]

# Extract the first elements of the variance vector for all time steps
var1 = [variance[k][1] for k in 1:T]
var2 = [variance[k][2] for k in 1:T]

var1_mpc = [variance_mpc[k][1] for k in 1:T]
var2_mpc = [variance_mpc[k][2] for k in 1:T]


# plot(1:T, var1, label="x1 variance", color=:blue, xlabel="Time Step", ylabel="Variance")
# plot!(1:T, var1_mpc, label="x1 variance MPC", color=:green, xlabel="Time Step", ylabel="Variance")

# plot(running_cost, label = "Stochastic Optimal Control", color = :blue, legend = :topright)
# plot!(running_cost_mpc, label = "MPC", color = :green, legend = :topright)

# Create a 2x2 layout for the subplots
plot_layout = @layout [a b; c d]


# Define consistent bin edges for estimation error histograms
num_bins = 21
min_estimation = minimum([minimum(est_err_rec), minimum(est_err_rec_mpc)])
max_estimation = maximum([maximum(est_err_rec), maximum(est_err_rec_mpc)])
bins_estimation = range(0, (max_estimation + (max_estimation - min_estimation) / (num_bins - 1)), length=num_bins)

# Define consistent bin edges for cost histograms
min_cost = minimum([minimum(cost_rec), minimum(cost_rec_mpc)])
max_cost = maximum([maximum(cost_rec), maximum(cost_rec_mpc)])
bins_cost = range(0, max_cost + (max_cost - min_cost) / (num_bins - 1), length=num_bins)  # Add one extra bin to the right

# Individual histograms with vertical dashed lines for means
p1 = histogram(est_err_rec, 
    bins=bins_estimation, 
    label="Stochastic Optimal Control", 
    color=:blue, 
    xlabel="Estimation Error", 
    ylabel="Frequency",
    legend=:topright)
vline!(p1, [mean_est_err_rec], label="Mean", color=:red, linestyle=:dash)

p2 = histogram(est_err_rec_mpc, 
    bins=bins_estimation, 
    label="Model Predictive Control", 
    color=:green, 
    xlabel="Estimation Error", 
    ylabel="Frequency",
    legend=:topright)
vline!(p2, [mean_est_err_rec_mpc], label="Mean", color=:red, linestyle=:dash)

p3 = histogram(cost_rec, 
    bins=bins_cost, 
    label="Stochastic Optimal Control", 
    color=:blue, 
    xlabel="Cost", 
    ylabel="Frequency")
vline!(p3, [mean_cost_rec], label="Mean", color=:red, linestyle=:dash)

p4 = histogram(cost_rec_mpc, 
    bins=bins_cost, 
    label="MPC", 
    color=:green, 
    xlabel="Cost", 
    ylabel="Frequency")
vline!(p4, [mean_cost_rec_mpc], label="Mean", color=:red, linestyle=:dash)
plot(p1, p2, p3, p4, layout=plot_layout, size=(800, 600))
# Combine the four histograms into a single plot

savefig("histograms.png")

