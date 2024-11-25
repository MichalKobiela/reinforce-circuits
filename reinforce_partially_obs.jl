using Distributions
using Zygote
using SpecialFunctions
using ProgressBars
using Flux 
using JLD2
include("model_mjp.jl")

num_unknowns_params = 4
num_design_params = 2
history_length = 1
response_length = 1

function loss_auto_paritally_obs(p,q, desired_oscillations_num = 5)
    sol = solution(vcat(p[1],q[1],q[2],q[3],p[2],q[4]))
    return (freq(autocorrelation(sol)) - 1/desired_oscillations_num).^2,freq(autocorrelation(sol))- 1/desired_oscillations_num
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
loss_vec = zeros(5000)
for i in tqdm(1:5000)
    unobserved_state = rand(4).*0.3 .+ 0.5
    state = zeros(num_design_params + response_length)
    for j in 1:history_length
        xs = sample_policy(state, model)
        loss_1, frequancy = loss_auto_paritally_obs(xs,unobserved_state)
        state = vcat(xs, frequancy*100)
    end

    xs = sample_policy(state, model)
    loss_vec[i] = loss_auto_paritally_obs(xs,unobserved_state)[1]
    ∇model= Flux.gradient(model -> score(xs, model, state) * loss_vec[i], model)[1]
    Flux.update!(opt_state, model, ∇model)
end


plot(-loss_vec, xlabel = "iteration", ylabel = "reward", legend = false) # plot progress

state = [2.0]
# result = sample_policy(state, model)
state = zeros(num_design_params + response_length)
result = sample_policy(state, model)

q = rand(4).*0.3 .+ 0.5
q[3] = 0.6
plot(solution(vcat(result[1],q[1],q[2],q[3],result[2],q[4])), xlabel = "time", ylabel = "expression", title = "first design", legend = false) # plot solution # plot solution

loss_1, frequancy = loss_auto_paritally_obs(result,q)
state = vcat(result, 100*frequancy)
result_2 = model(state)
loss_auto_paritally_obs(result_2,q)
plot(solution(vcat(result_2[1],q[1],q[2],q[3],result_2[2],q[4])), xlabel = "time", ylabel = "expression", title = "second design", legend = false) 

# range from 4 to 10
xs = 4:0.1:10

ys = [model([x])[4] for x in xs]
plot(xs, ys)



model_state = Flux.state(model)
jldsave("mymodel2.jld2"; model_state)

# Load the model
#model_state = JLD2.load("mymodel.jld2", "model_state");
#Flux.loadmodel!(model, model_state);