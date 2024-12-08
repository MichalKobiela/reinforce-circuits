using Catalyst, JumpProcesses, DifferentialEquations, ModelingToolkit, Plots
using DifferentialEquations




# Define variables (time-dependent now)
@variables t m₁(t) m₂(t) m₃(t) p₁(t) p₂(t) p₃(t)
@parameters α β δ γ K n

# Reactions for the repressilator
rxs = @reaction_network begin
    # Repression modeled with Hill functions
    (α / (1 + abs(p₃ / K)^n)), ∅ → m₁  # Repression of m₁ by p₃
    (α / (1 + abs(p₁ / K)^n)), ∅ → m₂  # Repression of m₂ by p₁
    (α / (1 + abs(p₂ / K)^n)), ∅ → m₃  # Repression of m₃ by p₂

    # Translation
    β, m₁ → p₁
    β, m₂ → p₂
    β, m₃ → p₃

    # Decay
    δ, m₁ → ∅
    δ, m₂ → ∅
    δ, m₃ → ∅
    γ, p₁ → ∅
    γ, p₂ → ∅
    γ, p₃ → ∅
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
    α => param_values[1]*1000,      # Maximal transcription rate
    β => param_values[2]*10,       # Translation rate
    δ => param_values[3],       # mRNA degradation rate
    γ => param_values[4],     # Protein degradation rate
    K => param_values[5]*10,       # Repression threshold
    n => param_values[6]*2+2       # Hill coefficient
]

@time jinput = JumpInputs(rxs, u0, tspan, params)

@time jprob = JumpProblem(jinput,save_positions = (false, false))

@time sol = solve(jprob, saveat = 0.1)

@time jprob= remake(jprob, p = param_values .* [1000, 10, 1, 1, 10, 2] + [0, 0, 0, 0, 0, 2])

plot(sol[4,250:1250])

function solution(param_values)
    
    global jprob= remake(jprob, p = param_values .* [1000, 10, 1, 1, 10, 2] + [0, 0, 0, 0, 0, 2])
    
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
    return 0
end

# Reward = - loss_auto
# Loss based on autocorrelation
function loss_auto(p, desired_oscillations_num = 5)
    d = 2*p[1]
    h = 3*p[2]+2
    sol = solution(p)
    return (freq(autocorrelation(sol)) - 1/desired_oscillations_num).^2
end

# Define the SDE problem
Sdesys = SDEProblem(rxs, u0, tspan, params)

using DifferentialEquations
# Solve the system
soluti = solve(Sdesys, EM(), dt = 0.1)

# Plot the solution
plot(soluti, vars=[p₁], xlabel="Time", ylabel="Protein Concentration",
     lw=2, label=["Protein1" "Protein2" "Protein3"], legend=:topright)

