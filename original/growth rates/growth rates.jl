# load in optical density data
# subset for each species
# view each subset and truncate to the portion for which a logistic growth curve is
# a good approximation e.g. trim death phase and/or secondary grwoth/biofilm formation.
# All species should be able to survive at this concentration, but if growth
# is not detectable on this plate reader for a given subset, the  odel wil not
# fit. Treat as zero.

# load packages
using Turing,  CSV, DataFrames,StatsPlots,Colors

# helper method to compute solution of logistic equation
sol(r ,k ,u ,t ) = (k * u) /(u + (k -u) * exp(-r * t) )
# take vector input (including estimated background optical density)
# compute solution + background
# return vector output
function sol(r ,k ,u ,t ,background )
    x=sol.(r',k',u,t) .+ background'
    return x#x)
end


#model

@model function hier_mod(y,t,n)

    R ~ truncated(Normal(0,1),0,2)
    σr ~ Exponential(1)
    r ~ filldist(truncated(Normal(R,σr),0,2),n)

    K ~ truncated(Normal(0,1),0,1)
    σk ~ Exponential(1)
    k ~ filldist(truncated(Normal(K,σk),0,1),n)

    u ~ truncated(Laplace(0,1),0,Inf)
    u *=0.01
    σ ~ Exponential(1)
    α ~ filldist(truncated(Normal(0,1),0,1),n)
    α .*= 0.1
    
    x = sol(r ,k ,u,t,α)
    y .~ Normal.(x, σ)
end

# load data
aopv = CSV.read("AOPV.csv", DataFrame)[(1:end .!= 18) .& (1:end .!= 85) .& (1:end .!= 113),2:end]
T = CSV.read("AOPV.csv", DataFrame)[(1:end .!= 18) .& (1:end .!= 85) .& (1:end .!= 113),1]

#Achromobacter
#subset data
a = hcat([aopv[: ,i] for i in 1:12:48]...)
ap = hcat([aopv[:,i] for i in 6:12:48]...)
at = hcat([aopv[:,i] for i in 49:12:96]...)
am = hcat([aopv[:,i] for i in 54:12:96]...)
# check for approximate consistency with logistic growth model
plot(a)
plot!(ap)
plot!(at)
plot!(am)
# trim where necesarry
a = a[1:110,:]
ap = ap[1:110,:]
at = at[1:110,:]
am = am[1:110,:]
#inference
amod = hier_mod(a,T[1:110],4) 
achn = sample(amod, NUTS(.65), MCMCThreads(), 1000, 4)
apchn = sample(apmod, NUTS(.65), MCMCThreads(), 1000, 4)
atmod = hier_mod(at,T[1:110],4) 
atchn = sample(atmod, NUTS(.65), MCMCThreads(), 1000, 4)
ammod = hier_mod(am,T[1:110],4) 
amchn = sample(ammod, NUTS(.65), MCMCThreads(), 1000, 4)


# Ochrobactrum

#subset data
o = hcat([aopv[: ,i] for i in 2:12:48]...)
op = hcat([aopv[:,i] for i in 7:12:48]...)
ot = hcat([aopv[:,i] for i in 50:12:96]...)
om = hcat([aopv[:,i] for i in 55:12:96]...)
# check for approximate consistency with logistic growth model
plot(o)
plot!(op)
plot!(ot)
plot!(om)
# trim where necesarry
o = o[1:130,:]
op = op[1:130,:]
om = om[1:130,:]

#inference
omod = hier_mod(o,T[1:130],4) 
ochn = sample(omod, NUTS(.65), MCMCThreads(), 1000, 4)
opmod = hier_mod(op,T[1:130],4) 
opchn = sample(opmod, NUTS(.65), MCMCThreads(), 1000, 4)
ommod = hier_mod(om,T[1:130],4) 
omchn = sample(ommod, NUTS(.65), MCMCThreads(), 1000, 4)

#Pseudomonas

#subset
p = hcat([aopv[1:44 ,i] for i in 3:12:48]...)
pp = hcat([aopv[1:64,i] for i in 8:12:48]...)
pt = hcat([aopv[1:64,i] for i in 51:12:96]...)
pm = hcat([aopv[1:69,i] for i in 56:12:96]...)
# check for approximate consistency with logistic growth model
plot(p)
plot!(pp)
plot!(pt)
plot!(pm)
# trim where necesarry
p = p[1:44,:]
pp = pp[1:64,:]
pm = pm[1:69,:]

#inference
pmod = hier_mod(p,T[1:44],4) 
pchn = sample(pmod, NUTS(.65), MCMCThreads(), 1000, 4)
ppmod = hier_mod(pp,T[1:64],4) 
ppchn = sample(ppmod, NUTS(.65), MCMCThreads(), 1000, 4)
pmmod = hier_mod(pm,T[1:69],4) 
pmchn = sample(pmmod, NUTS(.65), MCMCThreads(), 1000, 4)


# Variovorax
# subset data
v = hcat([aopv[: ,i] for i in 5:12:48]...)
v2 = hcat([aopv[: ,i] for i in 11:12:48]...)
vp = hcat([aopv[:,i] for i in 10:12:48]...)
vt = hcat([aopv[:,i] for i in 53:12:96]...)
vt2 = hcat([aopv[:,i] for i in 59:12:96]...)
vm = hcat([aopv[:,i] for i in 58:12:96]...)
v = hcat(v,v2)
vt = hcat(vt,vt2)
# check for approximate consistency with logistic growth model
plot(v)
plot!(vp)
plot!(vt)
plot!(vm)
# trim where necesarry

# inference
vmod = hier_mod(v,T[1:end],8) 
vchn = sample(vmod, NUTS(.65), MCMCThreads(), 1000, 4)
vpmod = hier_mod(vp,T[1:end],4) 
vpchn = sample(vpmod, NUTS(.65), MCMCThreads(), 1000, 4)
vtmod = hier_mod(vt,T[1:end],8) 
vtchn = sample(vtmod, NUTS(.65), MCMCThreads(), 1000, 4)
vmmod = hier_mod(vm,T[1:end],4) 
vmchn = sample(vmmod, NUTS(.65), MCMCThreads(), 1000, 4)


#Stenotrophomonas

#load and subset data
S = CSV.read("S.csv", DataFrame)[:,2:end]
ST = CSV.read("S.csv", DataFrame)[:,1]
n = 24
s = hcat([S[1:end,i+j] for i in 1:12:48 for j in 0:5 ]...,[S[:,i+j] for i in 7:12:96 for j in 0:5 ]...)
plot(s)

s = hcat([S[:,i+j] for i in 1:12:48 for j in 0:5 ]...)
sp = hcat([S[:,i+j] for i in 7:12:48 for j in 0:5 ]...)
st = hcat([S[:,i+j] for i in 49:12:96 for j in 0:5 ]...)
sm = hcat([S[:,i+j] for i in 55:12:96 for j in 0:5 ]...)
# check for approximate consistency with logistic growth model
plot(s)
plot!(sp)
plot!(st)
plot!(sm)
# trim where necesarry

#inference
smod = hier_mod(s,ST,24) 
schn = sample(smod, NUTS(.65), MCMCThreads(), 1000, 4)
spmod = hier_mod(sp,ST,24) 
spchn = sample(spmod, NUTS(.65), MCMCThreads(), 1000, 4)
smmod = hier_mod(sm,ST,24) 
smchn = sample(smmod, NUTS(.65), MCMCThreads(), 1000, 4)

# plotting
n = 4000
interactions = vcat(fill("parasite",n)...,fill("mutualist",n)...)
interaction = vcat(interactions...,interactions...,interactions...,interactions...,interactions...)
species = vcat(fill("Achromobacter sp.",2*n)...,fill("Ochrobactrum sp.",2*n)...,fill("Pseudomonas sp.",2*n)...,
fill("Stenotrophomonas sp.",2*n)...,fill("Variovorax sp.",2*n)... )
                
effect = vcat((apchn[:R] .- achn[:R])...,(amchn[:R] .- atchn[:R])...,
(opchn[:R] .- ochn[:R])...,(omchn[:R])...,
(ppchn[:R] .- pchn[:R])...,(pmchn[:R])...,
(spchn[:R] .- schn[:R])...,(smchn[:R])...,
(vpchn[:R] .- vchn[:R])...,(vmchn[:R] .- vtchn[:R])..., )

df = DataFrame([species,interaction,effect],[:species,:interaction,:effect])
paras = df[df[!,:interaction] .== "parasite",:]
muts = df[df[!,:interaction] .== "mutualist",:]

p1 =@df paras StatsPlots.violin(string.(:species), :effect,fillalpha = 0.8,color = cbPalette[2], side=:left, linecolor = cbPalette[4],trim = false,linewidth=1, label="1/64th TSB",tex_output_standalone = true,grid = false,legend =:topleft)
@df muts StatsPlots.violin!(string.(:species), :effect,fillalpha = 0.8,color = cbPalette[8], side=:right, linecolor = cbPalette[6],trim = false,
    ylab = "Change in intrinsic growth rate (r)",linewidth=1, label= "1/64th TSB \n0.2μg/ml tetracycline",tex_output_standalone = true,grid = false)
    StatsPlots.hline!([0],linestyle = :dash,linewidth = 1.5, color = :gray, label = false)
  
