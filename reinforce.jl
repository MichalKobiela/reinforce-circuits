using Distributions
using Zygote
using SpecialFunctions
using ProgressBars
using Flux 
include("model.jl")

desired_oscillations_num = 8

truncNorm(μ, b) = Truncated(Normal(μ, b), 0, 1)

function sample_policy(result)
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

function score(x, μ)
    σ = 0.025
    a = 0
    b = 1
    return log_truncated_normal_pdf_single(x[1], μ[1], σ, a, b) + log_truncated_normal_pdf_single(x[2], μ[2], σ, a, b) + log_truncated_normal_pdf_single(x[3], μ[3], σ, a, b) + log_truncated_normal_pdf_single(x[4], μ[4], σ, a, b)
end



result = [0.5,0.5,0.5,0.5] 

opt_state = Flux.setup(Descent(0.01), result)

loss_vec = zeros(5000)

#Reinforce
for i in tqdm(1:5000)
    xs = sample_policy(result)
    loss_vec[i] = loss_auto(xs,desired_oscillations_num)

    ∇μ= Flux.gradient(μ -> score(xs, μ), result)[1]
    Flux.update!(opt_state, result, ∇μ .* loss_vec[i])
end

plot(loss_vec) # plot progress

plot(solution(result)) # plot solution
