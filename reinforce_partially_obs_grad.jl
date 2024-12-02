using Distributions
using Zygote
using SpecialFunctions
using ProgressBars
using Flux 
using JLD2
using Random
include("model_mjp.jl")

num_unknowns_params = 4
num_design_params = 2
history_length = 1
response_length = 1

function loss_auto_paritally_obs(p,q, desired_oscillations_num = 5)
    sol = solution(vcat(p[1],q[1],q[2],q[3],p[2],q[4]))
    return (freq(autocorrelation(sol)) - 1/desired_oscillations_num).^2, freq(autocorrelation(sol))- 1/desired_oscillations_num
end


model = Chain(
  Dense((num_design_params + response_length) * history_length => 32, relu), 
  Dense(32 => 32, relu),
  Dense(32 => 32, relu),
  Dense(32 => num_design_params),
  sigmoid)

truncNorm(μ, b) = Truncated(Normal(μ, b), 0, 1) # Ensure samples are bounded

function sample_policy(state, model)
    result = model(state)
    x1 = rand(truncNorm(result[1],0.025))
    x2 = rand(truncNorm(result[2],0.025))
    return [x1,x2]
end

function log_truncated_normal_pdf_single(x::Real, μ::Real, σ::Real, a::Real, b::Real)
    if x < a || x > b
        return -Inf  # log(0) is -Inf
    else
        Φ_a = cdf(Normal(μ, σ), a)
        Φ_b = cdf(Normal(μ, σ), b)
        log_pdf_normal = logpdf(Normal(μ, σ), x)
        log_denominator = log(Φ_b - Φ_a + eps())  # Adding eps() for numerical stability
        return log_pdf_normal - log_denominator
    end
end

function score(x, model, state)
    μ = model(state)

    σ = 0.025
    a = 0
    b = 1
    return log_truncated_normal_pdf_single(x[1], μ[1], σ, a, b) + log_truncated_normal_pdf_single(x[2], μ[2], σ, a, b)
end


opt_state = Flux.setup(Descent(0.01), model)

#Reinforce
Random.seed!(0)

model = Chain(
  Dense((num_design_params + response_length) * history_length => 32, relu), 
  Dense(32 => 32, relu),
  Dense(32 => 32, relu),
  Dense(32 => num_design_params),
  sigmoid)

loss_vec = zeros(5000)
for i in tqdm(1:5000)
    unobserved_state = rand(4).*0.2 .+ 0.5
    initial_state = zeros(num_design_params + response_length)
    initial_xs = zeros(2)
    for j in 1:history_length
        initial_xs = sample_policy(initial_state, model)
        loss_1, frequancy = loss_auto_paritally_obs(initial_xs,unobserved_state)
        state = vcat(initial_xs, frequancy*100)
    end

    xs = sample_policy(state, model)
    loss_vec[i] = loss_1 + loss_auto_paritally_obs(xs,unobserved_state)[1]
    
    if loss_1 > 0.05
        loss_1 = 0.05
    end
    
    ∇model= Flux.gradient(model -> score(initial_xs, model, initial_state) * loss_1, model)[1]
    Flux.update!(opt_state, model, ∇model)

    if loss_vec[i] > 0.05
        loss_vec[i] = 0.05
    end

    ∇model= Flux.gradient(model -> score(xs, model, state) * loss_vec[i], model)[1]
    Flux.update!(opt_state, model, ∇model)
end

# 5.0k/5.0k [03:22<00:00, 25it/s]

plot(-loss_vec[1:5000], xlabel = "iteration", ylabel = "reward", legend = false) # plot progress

state = [2.0]
# result = sample_policy(state, model)
state = zeros(num_design_params + response_length)
result = sample_policy(state, model)

q = rand(4).*0.3 .+ 0.5
q[3] = 0.5
plot(solution(vcat(result[1],q[1],q[2],q[3],result[2],q[4])), xlabel = "time", ylabel = "expression", title = "first design", legend = false) # plot solution # plot solution

loss_1, frequancy = loss_auto_paritally_obs(result,q)
state = vcat(result, 100*frequancy)
result_2 = model(state)
loss_auto_paritally_obs(result_2,q)
plot(solution(vcat(result_2[1],q[1],q[2],q[3],result_2[2],q[4])), xlabel = "time", ylabel = "expression", title = "second design", legend = false) 

# range from 4 to 10

design1_losses = zeros(100)
design2_losses = zeros(100)

for i = 1:100
    q = rand(4).*0.3 .+ 0.5

    state = zeros(num_design_params + response_length)
    # result = model_one_step(state)

    # loss_1, frequancy = loss_auto_paritally_obs(result,q)
    # design_losses_single_step[i] = loss_1

    result = model(state)
    
    loss_1, frequancy = loss_auto_paritally_obs(result,q)
    design1_losses[i] = loss_1


    state = vcat(result, 100*frequancy)
    result_2 = model(state)
    design2_losses[i] = loss_auto_paritally_obs(result_2,q)[1]
end


model_one_step = Chain(
  Dense((num_design_params + response_length) * history_length => 32, relu), 
  Dense(32 => 32, relu),
  Dense(32 => 32, relu),
  Dense(32 => num_design_params),
  sigmoid)

loss_vec = zeros(5000)
for i in tqdm(1:5000)
    unobserved_state = rand(4).*0.2 .+ 0.5
    initial_state = zeros(num_design_params + response_length)
    initial_xs = zeros(2)
    for j in 1:history_length
        initial_xs = sample_policy(initial_state, model_one_step)
        loss_1, frequancy = loss_auto_paritally_obs(initial_xs,unobserved_state)
        state = vcat(initial_xs, frequancy*100)
    end

    xs = sample_policy(state, model_one_step)
    loss_vec[i] = loss_1 + loss_auto_paritally_obs(xs,unobserved_state)[1]
    
    if loss_1 > 0.05
        loss_1 = 0.05
    end
    
    ∇model= Flux.gradient(model_one_step -> score(initial_xs, model_one_step, initial_state) * loss_1, model_one_step)[1]
    Flux.update!(opt_state, model_one_step, ∇model)

    # if loss_vec[i] > 0.05
    #     loss_vec[i] = 0.05
    # end

    # ∇model= Flux.gradient(model -> score(xs, model, state) * loss_vec[i], model)[1]
    # Flux.update!(opt_state, model, ∇model)
end

design_losses_single_step = zeros(100)

for i = 1:100
    q = rand(4).*0.3 .+ 0.5
    loss_1, frequancy = loss_auto_paritally_obs(result,q)
    design_losses_single_step[i] = loss_1
end




plot(-design1_losses, xlabel = "trial", ylabel = "reward", label = "first design") # plot progress
plot!(-design2_losses, xlabel = "trial", ylabel = "reward", label = "second design") # plot progress

using Plots

# Sort the order in the circle by the second design
sorted_indices = sortperm(-design1_losses)

# Apply the sorted indices to both designs
sorted_design1_losses = design1_losses[sorted_indices]
sorted_design2_losses = design2_losses[sorted_indices]

# Create an array of angles from 0 to 2π
angles = range(0, stop=2π, length=length(sorted_design1_losses))

# Create the polar plot
logs = log.(design1_losses .+ 0.0001)
logs2 = log.(design2_losses .+ 0.0001)

minimum(logs)
minimum(logs2)


plot(-angles,  logs, xlabel = "trial", ylabel = "reward", label = "first design", proj = :polar) # plot progress
plot!(-angles,  logs2, xlabel = "trial", ylabel = "reward", label = "second design", proj = :polar) # plot progress
# Histograms
violin(-design1_losses, label = "first design") # plot progress
violin!(-design2_losses, label = "second design", xlabel = "", ylabel = "reward") # plot progress
violin!(-design_losses_single_step, label = "design single-step policy", xlabel = "", ylabel = "reward", legend = :bottom) # plot progress

ys = [model([x])[4] for x in xs]
plot(xs, ys)



# model_state = Flux.state(model)
# jldsave("mymodel4.jld2"; model_state)

# Load the model
model_state = JLD2.load("mymodel4.jld2", "model_state");
Flux.loadmodel!(model, model_state);