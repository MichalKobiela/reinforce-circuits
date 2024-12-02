using Catalyst, JumpProcesses, DifferentialEquations, ModelingToolkit, Plots
using DifferentialEquations




# Define variables (time-dependent now)
@variables t m₁(t) m₂(t) m₃(t) p₁(t) p₂(t) p₃(t)
@parameters α_1 α_2 α_3 β δ γ_1 γ_2 γ_3 K_1 K_2 K_3 n_1 n_2 n_3

# Reactions for the repressilator
rxs = @reaction_network begin
    # Repression modeled with Hill functions
    (α_1 / (1 + abs(p₃ / K_1)^n_1)), ∅ → m₁  # Repression of m₁ by p₃
    (α_2 / (1 + abs(p₁ / K_2)^n_2)), ∅ → m₂  # Repression of m₂ by p₁
    (α_3 / (1 + abs(p₂ / K_3)^n_3)), ∅ → m₃  # Repression of m₃ by p₂

    # Translation
    β, m₁ → p₁
    β, m₂ → p₂
    β, m₃ → p₃

    # Decay
    δ, m₁ → ∅
    δ, m₂ → ∅
    δ, m₃ → ∅
    γ_1, p₁ → ∅
    γ_2, p₂ → ∅
    γ_3, p₃ → ∅
end

# Initial conditions
u0 = [
    m₁ => 0.1, m₂ => 0.1, m₃ => 0.1,  # Initial mRNA concentrations
    p₁ => 1.0, p₂ => 1.0, p₃ => 1.0   # Initial protein concentrations
]

# Time span
tspan = (0.0, 125.0)

param_values = ones(6) .+ 0.5

params = [
    α_1 => param_values[1]*1000,      # Maximal transcription rate
    α_2 => param_values[2]*1000,      # Maximal transcription rate
    α_3 => param_values[3]*1000,      # Maximal transcription rate
    β => param_values[2]*10,       # Translation rate
    δ => param_values[3],       # mRNA degradation rate
    γ_1 => param_values[4],     # Protein degradation rate
    γ_2 => param_values[4],     # Protein degradation rate
    γ_3 => param_values[4],     # Protein degradation rate
    K => param_values[5]*100,       # Repression threshold
    n => param_values[6]*2+2       # Hill coefficient
]

@time jinput = JumpInputs(rxs, u0, tspan, params)

@time jprob = JumpProblem(jinput,save_positions = (false, false))

@time sol = solve(jprob, saveat = 0.1)

@time jprob= remake(jprob, p = param_values .* [1000, 10, 1, 1, 100, 2] + [0, 0, 0, 0, 0, 2])

plot(sol[4,250:1250])

function solution(param_values)
    
    global jprob= remake(jprob, p = param_values .* [1000, 10, 1, 1, 100, 2] + [0, 0, 0, 0, 0, 2])
    
    sol = solve(jprob, saveat = 0.1)
    
    return sol[4,250:1250]
end

@time solution(param_values)

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
    sol = solution(p)
    return (freq(autocorrelation(sol)) - 1/desired_oscillations_num).^2
end

# Function to find peaks in the autocorrelation
function peak(acf)
    for i in 2:length(acf)-1
        if acf[i] > acf[i-1] && acf[i] > acf[i+1]
            return acf[i]
        end
    end
    return -10
end

function loss_auto_peak(p, desired_oscillations_num)
    d = 2*p[1]
    h = 3*p[2]+2
    sol = solution(p)
    return - peak(autocorrelation(sol))
end

# Define the SDE problem
Sdesys = SDEProblem(rxs, u0, tspan, params)

using DifferentialEquations
# Solve the system
soluti = solve(Sdesys, EM(), dt = 0.1)

# Plot the solution
plot(soluti, vars=[p₁], xlabel="Time", ylabel="Protein Concentration",
     lw=2, label=["Protein1" "Protein2" "Protein3"], legend=:topright)

