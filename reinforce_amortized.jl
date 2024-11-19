using Distributions
using Zygote
using SpecialFunctions
using ProgressBars
using Flux 
using JLD2
include("model.jl")


model = Chain(
  Dense(1 => 16, relu),
  Dense(16 => 5),
  sigmoid)

truncNorm(μ, b) = Truncated(Normal(μ, b), 0, 1) # Ensure samples are bounded

function sample_policy(state, model)
    result = model(state)
    x1 = rand(truncNorm(result[1],0.025))
    x2 = rand(truncNorm(result[2],0.025))
    x3 = rand(truncNorm(result[3],0.025))
    x4 = rand(truncNorm(result[4],0.025))
    return [x1,x2,x3,x4]
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
    return log_truncated_normal_pdf_single(x[1], μ[1], σ, a, b) + log_truncated_normal_pdf_single(x[2], μ[2], σ, a, b) + log_truncated_normal_pdf_single(x[3], μ[3], σ, a, b) + log_truncated_normal_pdf_single(x[4], μ[4], σ, a, b)
end


opt_state = Flux.setup(Descent(0.1), model)

#Reinforce
loss_vec = zeros(30000)
for i in tqdm(1:30000)
    state = [rand([.4, .8])]
    xs = sample_policy(state, model)
    loss_vec[i] = loss_auto(xs,state[1] * 10)
    ∇model= Flux.gradient(model -> log_truncated_normal_pdf(xs, model, state) * loss_vec[i], model)[1]
    Flux.update!(opt_state, model, ∇model)
end

plot(loss_vec) # plot progress

state = [0.5]
# result = sample_policy(state, model)
result = model(state)
plot(solution(result)) # plot solution

# range from 4 to 10
xs = 4:0.1:10

ys = [model([x])[4] for x in xs]
plot(xs, ys)

model_state = Flux.state(model)
jldsave("mymodel.jld2"; model_state)

# Load the model
#model_state = JLD2.load("mymodel.jld2", "model_state");
#Flux.loadmodel!(model, model_state);