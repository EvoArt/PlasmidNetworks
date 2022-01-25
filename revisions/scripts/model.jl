using SymPy,CSV,DataFrames,Distributions, DifferentialEquations, LinearAlgebra, GLMakie,CairoMakie,
Chain, CategoricalArrays, StatsBase,DataFrameMacros,DiffEqGPU, JLD2
import SymPy.simplify

# Here I work with inequalities knowing that p is in the set (0,1)
# And all parameters are positive, since fitness and transmission cannot be negative,
# 0 transmission reduces to haploid selection model, with no internal equilibrium (unless ra = rb),
# and the model makes no sense if fitness is zero for one of the states.
SymPy.@syms p ra rb β
diffeq = p*(1-p)*((ra-rb)/((ra*p)+ rb*(1-p))) +p*(1-p)*β
solve(diffeq,p)
#1,0,(-ra - rb*β + rb)/(ra*β - rb*β)
pstar = (-ra - rb*β + rb)/(ra*β - rb*β)
simplify(pstar)
#-ra - rb⋅β + rb
#───────────────
#  β⋅(ra - rb)
# gives rb/(rb-ra) - 1/β 
pstar = rb/(rb-ra) - 1/β 
# interior equlibria exist when 0 < pstar <1
# 0 < rb/(rb-ra) - 1/β < 1
#1/β < rb/(rb-ra) < 1+ 1/β
#β > (rb-ra)/rb > β/(β+1)

recurs = p + p*(1-p)*((ra-rb)/((ra*p)+ rb*(1-p))) +p*(1-p)*β
d = simplify(diff(recurs,p))
e0 = simplify(d(p=>0))
#ra    
#── + β
#rb
# so (ra + βrb)/rb
# This is > 1 if ra + βrb > rb
#                     βrb > rb - ra
#                       β > (rb-ra)/rb
# This is true wherever internal equlibria exist: β > (rb-ra)/rb > β/(β+1)
e1 = simplify(d(p=>1))
#     rb
#-β + ──
#     ra
# so (rb - βra)/ra
# This is > 1 if rb - βra > ra
#                   - βra > ra - rb
#                     - β > (ra-rb)/ra
#                       β < (rb-ra)/ra
#
#       recall (rb-ra)/rb > β/(β+1)
#    (rb-ra)(β+1)/rb(β+1) > βrb/rb(β+1)
#         rb-ra +βrb -βra > βrb
#              rb-ra -βra > 0
#                     - β > (ra - rb)/ra
#                       β < (rb - ra)/ra

# Thus, whenever there are feasible internal equilibria,
# both external equilibria are unstable.
# So the internal equlibrium is globally stable.

# A more informative inequality for the bifurcation boundaries is:
# (rb-ra)/rb < β < (rb - ra)/ra

function multi(du,u,p,t)
    """
    Discrete time
    p = wa, wb, transmisssion matrix
    Diagonal elements of the transmission matrix are intraspecific,
    and each element i,j is the transmission rate to species i from species j 
    """
    transmission = vec((sum(p[3] .*u',dims = 2)  ) .* (1 .- u))
     du .= u .+ (u .*(1 .-u) .*(p[1] .-p[2])) ./ ((u .*p[1]) .+ ((1 .-u) .*p[2])) .+ transmission
end
function multiC(du,u,p,t)
    """
    Continuous time
    p = wa, wb, transmisssion matrix
    Diagonal elements of the transmission matrix are intraspecific,
    and each element i,j is the transmission rate to species i from species j 
    """
    transmission = vec((sum(p[3] .*u',dims = 2)  ) .* (1 .- u))
     du .= (u .*(1 .-u) .*(p[1] .-p[2])) .+ transmission
end

function ws(n ::Int64,nmut ::Int64)

    ra = rand(n)
    rb = rand(n)
    for i in 1:n
        if (sum(ra .> rb) < nmut) & (ra[i] < rb[i]) 
            ra[i], rb[i] = rb[i], ra[i]
        elseif (sum(ra .> rb) > nmut) & (ra[i] > rb[i]) 
            ra[i], rb[i] = rb[i], ra[i]
        end
    end
    return ra, rb
end

function betas(n,μ,σintra,σinter)
    """
    Returns transmission rate matrix with interspecific values drawn from half-normal N(0,σ)
    and intraspecific values twice as high on average.
    """
    #β = LowerTriangular(rand(truncated(Normal(μ,σinter),0,Inf),n,n))
    β = rand(truncated(Normal(σinter,σinter),0,Inf),n,n)
    β[diagind(β)] .= rand(truncated(Normal(σintra,σintra),0,Inf),n) #intraspecific transmission should be more frequent
    return β
end

function run_sim(nHost,nSim, t =1000.0)
    # prob = DiscreteProblem(multi,rand(nHost),(0.0,t), #this will be remade with new values each iteration
      prob = ODEProblem(multiC,rand(nHost),(0.0,t), #this will be remade with new values each iteration
     p = [rand(nHost),rand(nHost),rand(nHost,nHost),nHost])
     data = Array{Float64}(undef,nSim*(nHost+1),nHost +2) # creating an empty array and filling it is faster 
                              # than increasing the array size with each iteration
     Threads.@threads for m in 0:nHost # run simulations in parallel
          for row in (nSim*m+1):(nSim*(m+1))
             w = ws(nHost,m)
             M = sum(w[1] .> w[2])
             μ = rand(Uniform(0,0.05))
             σinter = rand(Uniform(0.0,0.05))
             σintra = σinter + rand(Uniform(0.0,0.05))
             β = betas(nHost,0.0,σintra,σinter) 
             prob = remake(prob,u0 = fill(0.001,nHost),
                                 p = [w..., β])
             hyperParams = [M,mean(β)]
             data[row,:] = vcat(hyperParams...,solve(prob,Tsit5()).u[end])
             
         end
     end
     return data
 end
 function run_simD(nHost,nSim, t =1000.0)
     prob = DiscreteProblem(multi,rand(nHost),(0.0,t), #this will be remade with new values each iteration
     p = [rand(nHost),rand(nHost),rand(nHost,nHost),nHost])
     data = Array{Float64}(undef,nSim*(nHost+1),nHost +2) # creating an empty array and filling it is faster 
                              # than increasing the array size with each iteration
     Threads.@threads for m in 0:nHost # run simulations in parallel
          for row in (nSim*m+1):(nSim*(m+1))
             w = ws(nHost,m)
             M = sum(w[1] .> w[2])
             μ = rand(Uniform(0,0.05))
             σinter = rand(Uniform(0.0,0.05))
             σintra = σinter + rand(Uniform(0.0,0.05))
             β = betas(nHost,0.0,σintra,σinter) 
             prob = remake(prob,u0 = fill(0.001,nHost),
                                 p = [w..., β])
             hyperParams = [M,mean(β)]
             data[row,:] = vcat(hyperParams...,solve(prob).u[end])
             
         end
     end
     return data
 end

 function shannon(a, n)
    A = sum(a)
    H = -sum([(a[i]/A) * log(a[i]/A) for i in 1:n])
    return isnan(H) ? 0 : H
end
G(a,n) = 2^(shannon(a,n))

function getResults(data, cutoff = 0.005, n = 10)
    M = data[:,1]
    β = data[:,2]
    x = data[:,3:end]
    x[x .< 0] .= 0
    C = sum(x .> cutoff, dims = 2) ./n
    G = 2 .^[shannon(row,n) for row in eachrow(x)]
    return DataFrame(hcat(M,β,C,G), [:M,:β,:C,:G])
end

data = run_sim(10,100,100000);
dfC = getResults(data,0.005);
CSV.write("results\\continuous100k.csv",dfC);
dataD = run_simD(10,100,100000);
dfD = getResults(dataD,0.005);
CSV.write("results\\discrete100k.csv",dfD);

#plots
β = collect(0:0.005:2)
ωA =  collect(0.0:0.005:1)
ωB = 1.0
p⃰(ωA,ωB,β) = ωB/(ωB-ωA) - 1/β
stab(ra,rb,β) = ( ra^2 *β + 2*ra^2  + ra*rb*β^2  - 4*ra*rb - rb^2 *β + 2*rb^2)/( ra^2  - 2*ra*rb + rb^2)

P = p⃰.(ωA',ωB,β)
for i in eachindex(P)
    if P[i] > 0
        if P[i] < 1.0
            P[i] = P[i]
        else
            P[i] = 1.0
        end
    else 
        P[i] = 0.0
    end
end
stabil = stab.(ωA',ωB,β) 
clamp!(stabil,-1,1)

# helper function for insets, from https://juliadatascience.io/makie_layouts
function add_axis_inset(; lab = "",pos=fig[1, 1], halign=0.1, valign=0.5,
    width=Relative(0.5), height=Relative(0.35), bgcolor=:lightgray)
    inset_box = Axis(pos, width=width, height=height,
        halign=halign, valign=valign, xlabel = lab,xlabelsize = 22,xticklabelsize=12, yticklabelsize=12,
        backgroundcolor=(:white,0.0))
    # bring content upfront
    translate!(inset_box.scene, 0, 0, 10)
    elements = keys(inset_box.elements)
    filtered = filter(ele -> ele != :xaxis && ele != :yaxis, elements)
    foreach(ele -> translate!(inset_box.elements[ele], 0, 0, 9), filtered)
    hideydecorations!(inset_box)
    hidexdecorations!(inset_box, label = false)
    hidespines!(inset_box)
    return inset_box
end

# helper function for cut() labels
f(from, to, i; leftclosed, rightclosed) = string(to)


GLMakie.activate!()
# Single host plot
β = collect(0:0.005:2)
fig = Figure(resolution = (935, 1200))
ax = Axis(fig[2:4, 1:2],ylabelsize = 30,xlabelsize = 30,ylabel = L"Relative fitness of infected hosts $R$", fontsize = 30,
backgroundcolor=(:white,0.0),title = "A", titlealign = :left,xlabel = L"Transmission rate $\beta$")
heat = contourf!(ax,β,ωA,P, levels = 0.0:0.1:1.0,
colormap = :Spectral, 
extendlow = :auto, extendhigh = :auto)
lines!(ax,β, 1 ./ (1 .+β) ,linewidth = 6,color = :black, label = L"\frac{\omega B - \omega A}{\omega B} < \beta < \frac{\omega B - \omega A}{\omega A} ", textsize = 30)
lines!(ax,0:0.1:1, 1 .- (0:0.1:1),linewidth = 6,color = :black) 
Colorbar(fig[1, 1:2], heat,label = L"Value of stable fixed point $p^{*}$", textsize = 30, vertical = false)
fig
hiax = add_axis_inset(;pos=fig[2:4, 1:2],lab ="",halign=0.60, valign=0.75 )
loax = add_axis_inset(;pos=fig[2:4, 1:2],lab ="",halign=-0.12, valign=0.08 )
text!(hiax,L"\beta > 1/R -1")
text!(loax,L"\beta < 1 - R")
 
# multiple host plots
df = CSV.read("results\\discrete100k.csv",DataFrame)

df = @subset(df, :β <0.06)
df[!,:β] = parse.(Float64,string.(cut(df.β, 0.00:0.006:0.06, labels = f)))

df2 = @chain df begin
    @groupby(:M, :β)
    @combine(:G = mean(:G),:C = mean(:C))
end

axG = Axis(fig[2, 5],title = "E",titlealign = :left)
gg = heatmap!(axG, df2.β,df2.M, df2.G, colormap = :Spectral,colorrange = (1,5))
Colorbar(fig[1, 5], gg, vertical = false,label = L"Generality $2^{H}$")
axC = Axis(fig[2, 6],title = "F",titlealign = :left)
cc = heatmap!(axC, df2.β,df2.M, df2.C, colormap = :Spectral,colorrange = (0,1))
Colorbar(fig[1, 6], cc, vertical = false,label = L"Connectance $\frac{\sum_{i=1}^{n} p_{i} > 0.005}{n}$")
df = CSV.read("results\\continuous100k.csv",DataFrame)
df = @subset(df, :β <0.06)

df[!,:β] = parse.(Float64,string.(cut(df.β, 0.00:0.006:0.06, labels = f)))

df2 = @chain df begin
    @groupby(:M, :β)
    @combine(:G = mean(:G),:C = mean(:C))
end

axGd = Axis(fig[3, 5],title = "G",titlealign = :left)
ggd = heatmap!(axGd, df2.β,df2.M, df2.G, colormap = :Spectral,colorrange = (1,5))
axCd = Axis(fig[3, 6],title = "H",titlealign = :left)
ccd = heatmap!(axCd, df2.β,df2.M, df2.C, colormap = :Spectral,colorrange = (0,1))

df = CSV.read("results\\gpu100k.csv",DataFrame)
df[!,:C] .= vec(sum(Array(df[:,12:21] .> 0.005) .& Array(df[:,2:11] .> 0.005), dims =2)) ./ vec(sum(Array(df[:,2:11] .> 0.005), dims =2)) 
df[!,:G] .=  [G(clamp.(Array(df[i,12:21])[(Array(df[i,12:21]) .> 0.005) .& (Array(df[i,2:11]) .> 0.005)],0.0,1.0),sum((Array(df[i,12:21]) .> 0.005) .& (Array(df[i,2:11]) .> 0.005))) for i in 1:size(df)[1]]
rename!(df, "x22" => "M","x1" => "β")
df[!,:dens] .= vec(sum(Array(df[:,2:11] .> 0.005), dims =2))

dfsub = @subset(df,:dens >9, :β <0.06)
dfsub[!,:β] = parse.(Float64,string.(cut(dfsub.β, 0.00:0.006:0.06, labels = f)))

df2 = @chain dfsub begin
    @groupby(:M, :β)
    @combine(:G = mean(:G),:C = mean(:C))
end


axGl = Axis(fig[4, 5],xticks = 0.012:0.024:0.06,title = "I",titlealign = :left)
ggl = heatmap!(axGl, df2.β,df2.M, df2.G, colormap = :Spectral,colorrange = (1,5))
axCl = Axis(fig[4, 6],xticks =  0.012:0.024:0.06,title = "J",titlealign = :left)
ccl = heatmap!(axCl, df2.β,df2.M, df2.C, colormap = :Spectral,colorrange = (0,1))

linkaxes!(axC,axCd,axCl,axG,axGd,axGl)
Ylab = Label(fig[2:4, 4],  L"no. mutualistic interactions $(R > 1)$", rotation = pi/2, textsize = 30)
Xlab = Label(fig[5, 5:6], L"Transmission rate $\beta$", textsize = 30, halign = :center,
    valign = :top,padding =(0, 0, 100, -50))
discinfo = Label(fig[2,7],  L"Discrete time $\Delta p$", rotation = 3pi/2)
continfo = Label(fig[3,7],  L"Continuous time $\frac{dp}{dt}$", rotation = 3pi/2)
discinfo = Label(fig[4,7],  L"Lotka-Volterra $\frac{dx}{dt}\frac{dp}{dt}$", rotation = 3pi/2)
hideydecorations!.([axC,axCd,axCl])
hidexdecorations!.([axC,axG,axCd,axGd])

# Streamplots
function f(x,y)
    u = [x,y]
    transmission = vec(sum(p[:,3:end] .*u',dims = 2 ) .* (1 .- u))
     du = (u .*(1 .-u) .*(p[:,1] .-p[:,2])) ./ ((u .*p[:,1]) .+ ((1 .-u) .*p[:,2])) .+ transmission
     return Point(du...)
end
  
para = Axis(fig[2,3],titlealign = :left,title = "B", xlabel = L"R = 0.9, β = 0.08,0.0", ylabel = L"R = 0.9, β = 0.09,0.0")
p = vcat([0.9,1.0,0.08,0.0]',[0.9,1.0,0.0,0.09]')
stplt = streamplot!(para, f, 0..1, 0..1,
    gridsize= (10,10), arrow_size = 8) 

trans = Axis(fig[3,3],titlealign = :left,title = "C", xlabel = L"R = 0.9, β = 0.08,0.02", ylabel = L"R = 0.9, β = 0.09,0.02")
p = vcat([0.9,1.0,0.08,0.02]',[0.9,1.0,0.02,0.09]')
stplt2 = streamplot!(trans, f, 0..1, 0..1,
    gridsize= (10,10), arrow_size = 8) 

mut = Axis(fig[4,3],titlealign = :left,title = "D", xlabel = L"R = 1.01, β = 0.08,0.02", ylabel = L"R = 0.9, β = 0.09,0.02")
p = vcat([1.01,1.0,0.08,0.02]',[0.9,1.0,0.02,0.09]')
stplt3 = streamplot!(mut, f, 0..1, 0..1,
    gridsize= (10,10), arrow_size = 8) 

hidedecorations!.([para,trans,mut], label = false)
 
fig
CairoMakie.activate!()
save("Fig1.pdf",fig)

GLMakie.activate!()
fig = Figure(resolution = (935, 1200))

df = CSV.read("results\\gpu100k.csv",DataFrame)
df[!,:C] .= vec(sum(Array(df[:,12:21] .> 0.005) .& Array(df[:,2:11] .> 0.005), dims =2)) ./ vec(sum(Array(df[:,2:11] .> 0.005), dims =2)) 
df[!,:G] .=  [G(clamp.(Array(df[i,12:21])[(Array(df[i,12:21]) .> 0.005) .& (Array(df[i,2:11]) .> 0.005)],0.0,1.0),sum((Array(df[i,12:21]) .> 0.005) .& (Array(df[i,2:11]) .> 0.005))) for i in 1:size(df)[1]]
rename!(df, "x22" => "M","x1" => "β")

dfsub = @subset(df,:β <0.06)
dfsub[!,:β] = parse.(Float64,string.(cut(dfsub.β, 0.00:0.006:0.06, labels = f)))

df2 = @chain dfsub begin
    @groupby(:M, :β)
    @combine(:G = mean(:G),:C = mean(:C))
end


axGl = Axis(fig[2, 1],xticks = 0.012:0.024:0.06,title = "A",titlealign = :left,
ylabel = L"no. mutualistic interactions $(R > 1)$", ylabelsize = 30)
ggl = heatmap!(axGl, df2.β,df2.M, df2.G, colormap = :Spectral,colorrange = (1,5))
axCl = Axis(fig[2, 2],xticks =  0.012:0.024:0.06,title = "B",titlealign = :left)
ccl = heatmap!(axCl, df2.β,df2.M, df2.C, colormap = :Spectral,colorrange = (0,1))

Colorbar(fig[1, 1], ggl, vertical = false,label = L"Generality $2^{H}$")
Colorbar(fig[1, 2], ccl, vertical = false,label = L"Connectance $\frac{\sum_{i=1}^{n} x_{i} > 0.005}{n}$")
Xlab = Label(fig[3, 1:2], L"Transmission rate $\beta$", textsize = 30, halign = :center,
    valign = :top,padding =(0, 0, 100, -10))

CairoMakie.activate!()
save("figS6.pdf",fig)
