using CSV,DataFrames,Distributions, DifferentialEquations, LinearAlgebra,StatsBase,DiffEqGPU,JLD2

function LV_comp(du,u,p,t)

    # parameters
    k = 1
    n = 10
    A = view(p,1:100)
    ra = view(p,101:110)
    rb = view(p,111:120)
    β = view(p,121:220)
    # host dynamics
    for i in 1:n
        du[i] = u[i] *(ra[i] * u[n+i] + rb[i] *(1 - u[n+1]))
        for j in 1:n
            du[i] -= u[i]*(ra[i] * u[n+i] + rb[i] *(1 - u[n+1]))*A[(i-1)*10+j]*u[j]/k
        end
    end
    # symbiont dynamics
    for i in 1:n
        du[n+i] = u[n+i] *(1 - u[n+i]) *(ra[i] -rb[i])
        for j in 1:n
            du[n+i] += β[(i-1)*10+j]*(1-u[n+i])*u[n+j]*u[j]
        end
    end
end

function get_pars(nspecies,c,nmut,σinter = 0.01,σintra=0.05, μe = 0,μa = 0)
    α = zeros(nspecies,nspecies)
    ra = rand(nspecies)
    rb = rand(nspecies)
    for i in 1:nspecies
        if (sum(ra .> rb) < nmut) & (ra[i] < rb[i]) 
            ra[i], rb[i] = rb[i], ra[i]
        elseif (sum(ra .> rb) > nmut) & (ra[i] > rb[i]) 
            ra[i], rb[i] = rb[i], ra[i]
        end
    end
    for i in 1:nspecies-1
        for j in i:nspecies
            if rand() < c
                α[i,j] = rand() 
                α[j,i] = rand()
            end
        end
    end
    α[diagind(α)] .= 1
    β = rand(truncated(Normal(μe,σinter),0,Inf),nspecies,nspecies)
    β[diagind(β)] .= rand(truncated(Normal(μa,σintra),0,Inf),nspecies) #intraspecific transmission should be more frequent

    return vcat(vec.([α,ra,rb,β])...)
end
p =get_pars(10,0.5,2)
u0 = rand(20)
pro = ODEProblem(LV_comp,u0,(0.0,1000.0),p)
function prob_fun(prob,i,repeat)
    c = rand()
        nmut = rand(0:10)
	    μe = rand(Uniform(0.0,0.05))
        σinter = μe
	    μa= μe + rand(Uniform(0.0,0.05))
        σintra= μa
        u0 = vcat(rand(10)...,rand(10) ./100...)
        p = get_pars(10,c,nmut,σinter,σintra,μe,μa)
    remake(prob, p = p, u0 = u0)  
end
montepro = EnsembleProblem(pro, prob_func = prob_fun, safetycopy=true)

sols = solve(montepro,Tsit5(),EnsembleGPUArray(),trajectories=100000,saveat = 1000.0)
save("gpu100k.jld2", Dict("100k" => sols))

k100 = load("gpu100k.jld2")["100k"]
u = vcat([x.u[end]' for x in k100]...)
β =[mean(x.prob.p[121:end]) for x in k100]
M =[sum(x.prob.p[101:110] .> x.prob.p[111:120]) for x in k100]
res = hcat(β,u,M)
CSV.write("gpu100k.csv",DataFrame(res,:auto))
