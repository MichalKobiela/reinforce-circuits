# Run following code to install required julia packages
# using Pkg
# Pkg.add("DifferentialEquations")
# Pkg.add("Statistics")
# Pkg.add("Plots")
# Pkg.add("FFTW")
# Pkg.add("BenchmarkTools")
# Pkg.add("StatsBase")
# Pkg.add("Random")
# Pkg.add("DataFrames")

using StochasticDiffEq
using Statistics
using Plots
using FFTW
using BenchmarkTools
using StatsBase
using Random
using DataFrames


Random.seed!(0)

k_transcription = 100.0
k_degradation_A = 0.5
k_degradation_B = 0.5
k_degradation_C = 0.5
n_A = 2.1
n_B = 2.0
n_C = 1.9
K_A = 1.0
K_B = 1.0
K_C = 1.0



p = [0.5, 0.5, 0.5, 2.1, 2.0, 1.9, 100.0, 1.0]

ground_truth_unc = [0.5, 0.5, 0.5, 2.0, 2.0, 2.0]

function f(du, u, p, t)
    A = u[1]
    B = u[2] 
    C = u[3]

    k_degradation_A = p[1]
    k_degradation_B = p[2]
    k_degradation_C = p[3]
    n_A = p[4]
    n_B = p[5]
    n_C = p[6]
    k_transcription = p[7]
    K = p[8]
    
    du[1] = k_transcription / (1 + abs(C / K)^n_C) - k_degradation_A*A
    du[2] = k_transcription / (1 + abs(A / K)^n_A) - k_degradation_B*B
    du[3] = k_transcription / (1 + abs(B / K)^n_B) - k_degradation_C*C
end



function g(du, u, p, t)
    A = u[1]
    B = u[2] 
    C = u[3]

    k_degradation_A = p[1]
    k_degradation_B = p[2]
    k_degradation_C = p[3]
    n_A = p[4]
    n_B = p[5]
    n_C = p[6]
    k_transcription = p[7]
    K = p[8]

    du[1,1] = sqrt(abs(k_transcription / (1 + abs(C / K)^n_C)))
    du[1,2] = sqrt(abs(k_degradation_A*A))
    du[1,3] = 0 
    du[1,4] = 0
    du[1,5] = 0
    du[1,6] = 0

    du[2,1] = 0
    du[2,2] = 0
    du[2,3] = sqrt(abs(k_transcription / (1 + abs(A / K)^n_A))) 
    du[2,4] = sqrt(abs(k_degradation_B*B))
    du[2,5] = 0
    du[2,6] = 0

    du[3,1] = 0
    du[3,2] = 0
    du[3,3] = 0
    du[3,4] = 0
    du[3,5] = sqrt(abs(k_transcription / (1 + abs(B / K)^n_B)))
    du[3,6] = sqrt(abs(k_degradation_C*C))
end

init_cond = zeros(3)
prob = SDEProblem(f, g,init_cond, (0.0, 200.0),p,noise_rate_prototype = zeros(3, 6)) 

function solution(p)
    d = 2*p[1]
    h = 3*p[2]+2
    return solve(prob, EM(), dt = 0.1, p = [d,d,d,h,h,h,p[3]*1000,p[4]*10])[2,1000:2000]
end

# Function to compute autocorrelation of a signal
function autocorrelation(signal)
    n = length(signal)
    acf = zeros(Float64, n)
    for lag in 0:n-1
        acf[lag+1] = sum(signal[1:n-lag] .* signal[lag+1:n])
    end
    return acf / maximum(acf)
end

# Function to find peaks in the autocorrelation
function freq(acf)
    for i in 2:length(acf)-1
        if acf[i] > acf[i-1] && acf[i] > acf[i+1]
            return float(i/1000)
        end
    end
    return -10
end

# Reward = - loss_auto
# Loss based on autocorrelation
function loss_auto(p, desired_oscillations_num = 5)
    d = 2*p[1]
    h = 3*p[2]+2
    sol = solve(prob, EM(), dt = 0.1, p = [d,d,d,h,h,h,1000*p[3],10*p[4]])[2,1000:2000]
    return (freq(autocorrelation(sol)) - 1/desired_oscillations_num).^2
end

function loss_auto_partially_obs(p,q, desired_oscillations_num = 5)

    d1 = 2*p[1]
    d2 = 2*p[2]
    d3 = 2*p[3]
    h1 = 3*p[1]+2
    h2 = 3*p[2]+2
    h3 = 3*p[3]+2
    sol = solve(prob, EM(), dt = 0.1, p = [d1,d2,d3,h1,h2,h3,1000*q[1],10*q[2]])[2,1000:2000]
    return (freq(autocorrelation(sol)) - 1/desired_oscillations_num).^2
end

function solution_partially_obs(p,q)
    d = 2*q[3]
    h1 = 2*p[1]+1
    h2 = 2*p[2]+1
    h3 = 2*p[3]+1
    return solve(prob, EM(), dt = 0.1, p = [d,d,d,h1,h2,h3,1000*q[1],100*q[2]])[2,1000:2000]
end
