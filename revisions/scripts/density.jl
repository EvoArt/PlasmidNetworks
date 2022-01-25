# In response to reviewer 2, comment 2:

# As the authors mention, the tetracycline treatment will also have different impacts on the 
# bacterial populations, with some more resistant than others. Could the result be understood simply 
# in terms of the change in population densities of the other members of the community (rather than 
# the switch in sign from mutualist to parasite for the plasmid)? If overall population densities are 
# lower when tetracycline is applied, presumably the plasmid would transmit less efficiently, which 
# may yield a comparable result. 

# In this script I want to confirm whenther or not total densities were impacted by tetracycline, in the
# presence of pKJK5

# Load packages and data
using DataFrames,CSV, Turing, GLMakie, AlgebraOfGraphics, Chain, DataFrameMacros, CategoricalArrays,
KernelDensity,StatsFuns,JLD2, CairoMakie
import GLMakie.density

include("scripts\\helpers.jl")
df = CSV.read("data\\speciescountdata.csv",DataFrame)
df .= ifelse.(df .== "NA", "0", df)
checks = Array(parse.(Int,df[:,17:26]))
props = hcat([checks[:,i] ./checks[:,i-1] for i in 2:2:10]...)
df[!,:od] = df.OD .* props[:,1]
df[!,:pc] = df.PC .* props[:,2]
df[!,:aa] =  df.AA .* props[:,3]
df[!,:sr] = df.SR .* props[:,4]
df[!,:vg] = df.VG .* props[:,5]
df.trt = levelcode.(CategoricalArray(df[!,"treat.2"]))
@chain df begin
    @transform!(:C = sum([:aa,:od,:pc,:sr,:vg] .>0))
    @transform!(:G = 2^shannon([:aa,:od,:pc,:sr,:vg],:C))
end
# Look at total colony counts, separated by week and treatment
plt = data(df) * mapping(:treatment, :total, row=:week => nonnumeric, color = "treat.2")
draw(plt)
# In weeks 1 and 3, tetracycline leads to reduced density in the presence of the plasmid.
# But in week 5, the opposite is true.
# Since generality and connectance are higher in the presence of tetracycline in both
# week 1 and 5, it is unlikely that total cell densities explain this pattern.

# It may be worth investigating whether cell density predicts generality/connectance within
# week/treatment. However, there is a potentially large confounding factor here. Higher colony counts
# increase the chances of detecting rare transconjugants. This may cause bias in favour of a positive
# relationship between cell density and both network metrics.

# Specify regression models

# Normal model for generality
@model function density_G(tot,trt,G)
    # Hyper priors
    μ ~ Normal(0,1)
    σ ~ Exponential(1)
    # Priors
    β ~ filldist(Normal(μ,σ), 6)
    α ~ filldist(Normal(0,4),6)
    ϵ ~ Exponential(1)
    # Likelihood
    Y = α[trt] .+ β[trt] .* tot
    G .~ Normal.(Y,ϵ)
end
# Poisson model for number of links (out of 5)
@model function density_C(tot,trt,C)
    # Hyper priors
    μ ~ Normal(0,1)
    σ ~ Exponential(1)
    # Priors
    β ~ filldist(Normal(μ,σ), 6)
    α ~ filldist(Normal(0,1),6)
    # Likelihood
    p = logistic.(α[trt] .+ β[trt] .* tot)
    C .~ Binomial.(5,p)
end
# Subset data
plas = @subset(df, :plasmid =="yes")
plas[!,:trt] = levelcode.(CategoricalArray(plas[!,"treat.2"]))
plas[!,:tet] = levelcode.(CategoricalArray(plas[!,"tet"]))
week1 = @subset(plas, :week ==1)
week5 = @subset(plas, :week ==5)
# Sample from posterior
Gchn1 = sample(density_G(week1.total,week1.trt,week1.G),NUTS(0.95),MCMCThreads(),5000,4)
Gchn5 = sample(density_G(week5.total,week5.trt,week5.G),NUTS(0.95),MCMCThreads(),5000,4)
Cchn1 = sample(density_C(week1.total,week1.trt,week1.C),NUTS(0.95),MCMCThreads(),5000,4)
Cchn5 = sample(density_C(week5.total,week5.trt,week5.C),NUTS(0.95),MCMCThreads(),5000,4)
# Output results
CSV.write("results\\density_G1.csv",DataFrame(Gchn1))
CSV.write("results\\density_G5.csv",DataFrame(Gchn5))
CSV.write("results\\density_C1.csv",DataFrame(Cchn1))
CSV.write("results\\density_C5.csv",DataFrame(Cchn5))

# We found little evidence of any relationship (positive or negative) between cell density
# and either network metirc in week 1 or week 5.

#Plot
G1 = CSV.read("results/density_G1.csv",DataFrame).μ
G5 = CSV.read("results/density_G5.csv",DataFrame).μ
C1 = CSV.read("results/density_C1.csv",DataFrame).μ
C5 = CSV.read("results/density_C5.csv",DataFrame).μ

GLMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1],xlabelsize = 20, xlabel = "Regression coefficient hyperparameter estimate")
g_1 = kde(G1)
g_5 = kde(G5)
c_1 = kde(C1)
c_5 = kde(C5)

band!(ax, c_5.x,fill(1.5,length(c_5.density)),1.5 .+ c_5.density ./ maximum(c_5.density), color = (:red,0.4))
q5 = findfirst(c_5.x .> quantile(C5,0.05))
q95 = findfirst(c_5.x .> quantile(C5,0.95))
band!(ax, c_5.x[q5:q95],fill(1.5,length(c_5.density[q5:q95])),1.5 .+ c_5.density[q5:q95] ./ maximum(c_5.density), color = (:red,0.4))

band!(ax, c_1.x,ones(length(c_1.density)),1 .+ c_1.density ./ maximum(c_1.density), color = (:red,0.4))
q5 = findfirst(c_1.x .> quantile(C1,0.05))
q95 = findfirst(c_1.x .> quantile(C1,0.95))
band!(ax, c_1.x[q5:q95],fill(1,length(c_1.density[q5:q95])),1 .+ c_1.density[q5:q95] ./ maximum(c_1.density), color = (:red,0.4))

band!(ax, g_5.x,fill(0.5,length(g_5.density)),0.5 .+ g_5.density ./ maximum(g_5.density), color = (:blue,0.4))
q5 = findfirst(g_5.x .> quantile(G5,0.05))
q95 = findfirst(g_5.x .> quantile(G5,0.95))
band!(ax, g_5.x[q5:q95],fill(0.5,length(g_5.density[q5:q95])),0.5 .+ g_5.density[q5:q95] ./ maximum(g_5.density), color = (:blue,0.4))

band!(ax, g_1.x,zeros(length(g_1.density)),g_1.density ./ maximum(g_1.density), color = (:blue,0.4))
q5 = findfirst(g_1.x .> quantile(G1,0.05))
q95 = findfirst(g_1.x .> quantile(G1,0.95))
band!(ax, g_1.x[q5:q95],fill(0.0,length(g_1.density[q5:q95])), g_1.density[q5:q95] ./ maximum(g_1.density), color = (:blue,0.4))

vlines!(ax,0, linestyle = :dash, color = :black, linewidth = 1.5)
ax.yticks = ([0,0.5,1,1.5], ["Generality week 1","Generality week 5", "Connectance week 1", "Connectance week 5"])
hidedecorations!(ax, label = false, ticklabels = false, ticks = false)

CairoMakie.activate!()
save("dens.pdf",fig)


