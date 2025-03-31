using Revise
using Plots, LinearAlgebra
#import Pkg; Pkg.add("Zygote"); Pkg.add("Convex"); Pkg.add("SCS")
using Zygote
using Convex, SCS
using JLD2

##################################################

include("./src/eKF.jl")
include("./src/control_sampler.jl")
include("./src/mpc.jl")
include("./simulate.jl")


"""
Initialize battery dynamics and measurement model.

"""
n = 6 # Number of states
m = 6 # Number of inputs
η = 1 # Efficiency
Q_nom = 2.2 # Nominal capacity
dt = 1.0 # Discretization time
A = I(n)
B = dt * η / Q_nom * I(m)


function state_dynamics(SOC, I)
    SOC = A * SOC .+ B * I
    SOC = clamp.(SOC, 0.0, 1.0)
    return SOC
end

function measurement_dynamics(SOC)
    # LTO
    OCV_LTO = 2.5 + 0.3 * SOC[1] + 0.1 * tanh(8 * (SOC[1] - 0.5)) + 0.05 * sin(8 * π * SOC[1])   
    # LCO
    OCV_LCO = 3.7 + 0.5 * SOC[2] + 0.3 * sin(2 * π * SOC[2])

    # Li=ion from Hao Wang et al. 2021,
    "Lithium-Ion Battery SOC Estimation Based on Adaptive Forgetting Factor Least Squares Online Identification and Unscented Kalman Filter"
    OCV_Li = -43.1 * SOC[3]^6 + 155.4 * SOC[3]^5 - 215.7 * SOC[3]^4 + 146.6 * SOC[3]^3 - 50.16 * SOC[3]^2 + 8.674 * SOC[3] + 2.991

    # Li-ion from Sundaresan et al. 2022,
    "Tabular Open Circuit Voltage Modelling of Li-Ion Batteries for Robust SOC Estimation"
    # Li-ion exponential model
    OCV_EXP = 3.679*exp(-0.1101*SOC[4]) - 0.2528*exp(-6.829*SOC[4]) + 0.9386*SOC[4]^2
    # Li-ionsum of sines model
    OCV_SIN = 4.848*sin(1.512*SOC[4] + 0.5841) + 7.715*sin(4.756*SOC[4] + 1.99) + 6.655 * sin(4.928*SOC[4] + 5.038)

    # Li-ion from Woo-Yong Kim et al. 2019,
    "A Nonlinear-Model-Based Observer for a State-of-Charge Estimation of a Lithium-Ion Battery in Electric Vehicles"

    # OCV_Kim = 0.9878 + 0.32095*SOC[5] 
    # + 0.07*sin(1.90*SOC[5]-3.30) 
    # + 0.05*sin(0.3*SOC[5]+0.49) 
    # + 0.04*sin(3.39*SOC[5]-0.98)
    # + 0.02*sin(8.35*SOC[5]-1.27)
    # + 0.23*sin(10.01*SOC[5]+1.74) 
    # + 0.22*sin(10.10*SOC[5]-1.42)

    # Li-ion polynomial model from Zhao et al. 
    OCV_Zhao = 3.151 + 0.401*SOC[5] + 4.1410*SOC[5]^2 + -26.228*SOC[5]^3 + 57.835*SOC[5]^4 - 55.688*SOC[5]^5 + 19.808*SOC[5]^6

    return [OCV_LTO; OCV_LCO; OCV_Li; OCV_EXP; OCV_SIN; OCV_Zhao]
end

"""
Initialize extended Kalman filter

"""
W = 0.01 * I(n)
V = 0.01 * I(m)
eKF = ExtendedKalmanFilter(state_dynamics, measurement_dynamics, W, V)


"""
Initialize the control sampler

"""
N = 6 # prediction horizon length
Q = 1.0*I(n)
R = 0.1 * I(m)
set_point = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
running_cost = (x, cov, u) -> (x-set_point)' * Q * (x-set_point)  + tr(Q*cov) + u' * R * u
# running_cost = (x, cov, u) -> Q[1,1]*(sum(x)-set_point)^2  + tr(Q*cov) + u' * R * u
CS = ControlSampler(eKF, N, running_cost)

"""
Initialize the simulation parameters:

L: number of candidate control trajectories
T: number of time steps
x₀₀: initial state
Σ₀₀: initial covariance matrix

"""
x₀₀ = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1]
Σ₀₀ = 0.1 * Matrix{Float64}(I, n, n)
L = 100
T = 50
num_simulations = 500

# for recording results
cost_rec = zeros(num_simulations)
est_err_rec = zeros(num_simulations)
x_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations) 
u_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations) 
x_true_rec = Vector{Vector{Vector{Float64}}}(undef, num_simulations)  
cov_rec = Vector{Vector{Matrix{Float64}}}(undef, num_simulations) 


"""
Run stochastic optimal control simulation

"""

function simulation_run()
    X_rec, U_rec, Σ_rec, X_true_rec = simulate_CS(x₀₀, Σ₀₀, T, L; u_noise_cov = 0.01)
    achieved_cost = sum([CS.running_cost(X_true_rec[k], 0, U_rec[k]) for k in 1:T]) / T
    achieved_est_err = sum([norm(X_rec[k] - X_true_rec[k]) for k in 1:T]) / T
    return achieved_cost, achieved_est_err, X_rec, U_rec, Σ_rec, X_true_rec
end

for i in 1:num_simulations
    println("Simulation: ", i)
    if i==num_simulations
        @time begin
        achieved_cost, achieved_est_err, X_rec, U_rec, Σ_rec, X_true_rec = simulation_run()
        end
    else
        achieved_cost, achieved_est_err, X_rec, U_rec, Σ_rec, X_true_rec = simulation_run()
    end

    cost_rec[i] = achieved_cost
    est_err_rec[i] = achieved_est_err
    x_rec[i] = X_rec
    u_rec[i] = U_rec
    x_true_rec[i] = X_true_rec
    cov_rec[i] = Σ_rec
end

println("Average Achieved Cost: ", sum(cost_rec) / num_simulations)
println("Average Achieved Estimation Error: ", sum(est_err_rec) / num_simulations)


"""
X_rec = reduce(hcat, X_rec)
U_rec = reduce(hcat, U_rec)
X_true_rec = reduce(hcat, X_true_rec)
variances = [[Σ_rec[i][1,1] ;Σ_rec[i][2,2]] for i in 1:T]
variances = reduce(hcat, variances)
"""

# for recording results
cost_rec_mpc = zeros(num_simulations)
est_err_rec_mpc = zeros(num_simulations)
x_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations) 
u_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations) 
x_true_rec_mpc = Vector{Vector{Vector{Float64}}}(undef, num_simulations)  
cov_rec_mpc = Vector{Vector{Matrix{Float64}}}(undef, num_simulations) 

"""
Run MPC simulation

"""
function simulate_run_mpc()
    X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc = simulate_mpc(x₀₀, Σ₀₀, T)
    achieved_cost = sum([CS.running_cost(X_true_rec_mpc[k], 0, U_rec_mpc[k]) for k in 1:T]) / T
    achieved_est_err = sum([norm(X_rec_mpc[k] - X_true_rec_mpc[k]) for k in 1:T]) / T
    return achieved_cost, achieved_est_err, X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc
end

for i in 1:num_simulations
    achieved_cost, achieved_est_err, X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc = simulate_run_mpc()
    cost_rec_mpc[i] = achieved_cost
    est_err_rec_mpc[i] = achieved_est_err
    x_rec_mpc[i] = X_rec_mpc
    u_rec_mpc[i] = U_rec_mpc
    x_true_rec_mpc[i] = X_true_rec_mpc
    cov_rec_mpc[i] = Σ_rec_mpc
end

println("Average Achieved Cost: ", sum(cost_rec_mpc) / num_simulations)
println("Average Achieved Estimation Error: ", sum(est_err_rec_mpc) / num_simulations)

"""
X_rec_mpc, U_rec_mpc, Σ_rec_mpc, X_true_rec_mpc = simulate_mpc(x₀₀, Σ₀₀, T)
X_rec_mpc = reduce(hcat, X_rec_mpc)
U_rec_mpc = reduce(hcat, U_rec_mpc)
X_true_rec_mpc = reduce(hcat, X_true_rec_mpc)
variances_mpc = [[Σ_rec_mpc[i][1,1] ;Σ_rec_mpc[i][2,2]] for i in 1:T]
variances_mpc = reduce(hcat, variances_mpc)
"""
##################################################
println("Average Achieved Cost Change: % ", (sum(cost_rec)-sum(cost_rec_mpc)) / sum(cost_rec_mpc)*100)
println("Average Achieved Estimation Error Change: % ", (sum(est_err_rec) - sum(est_err_rec_mpc)) / sum(est_err_rec_mpc)*100)

##################################################


SAVE_DATA = false
if(SAVE_DATA)
    @save "simulation_results.jld2" x_rec u_rec cov_rec x_true_rec cost_rec est_err_rec x_rec_mpc u_rec_mpc cov_rec_mpc x_true_rec_mpc cost_rec_mpc est_err_rec_mpc
end
