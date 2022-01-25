using Turing, Images, ImageSegmentation
using  ImageContrastAdjustment,FileIO
mi =8.3167 - 6.133262
ma =8.3167 + 6.133262
	hrdat4 = Dict()
		path_three = "D:\\Arthur N\\UG conjugation assay\\3 hours\\10x 38.3ms 50\\"
	i = 1
	trt = ""
	pic_path = readdir(p)[3]
	for i in [1,3,4,5,6]
		for trt in ["t",""]
			p = path_three * trt *string(i) * "\\"	
			dat  = []		
			for pic_path in readdir(p)
				pic = load(p*pic_path)
				graycut = 0.15
				print(i)
				segs = fast_scanning((Gray.(pic) .> graycut),0.1)
					counts = segment_pixel_count(segs)
					print(i)
					kv = [k for (k,v) in counts if (v >mi) & (v <ma)]
					print(i)
	
				push!(dat,length(kv))
				hrdat4[trt *string(i)] = dat
				print(i)
			end
		end
	end
	
	
		for trt in ["2","3","t2"]
		
			p = path_three * trt * "\\"
			
			dat  = []
			
			for pic_path in readdir(p)
				pic = load(p*pic_path)
				
				graycut = 0.2251
				segs = fast_scanning( (Gray.(pic) .> graycut),0.1)
					counts = segment_pixel_count(segs)
					kv = [k for (k,v) in counts  if (v >mi) & (v <ma)]
					
				push!(dat,length(kv))
				hrdat4[trt ] = dat
			end
		end

control = hcat([hrdat4[string(i)] for i in 1:6]...)
controldf = DataFrame( [vcat([fill(i,6) for i in 1:6]...), vec(control)], [:rep,:count])
tet = hcat([hrdat4["t" *string(i)] for i in 1:6]...)
tetdf = DataFrame( [vcat([fill(i,6) for i in 7:12]...), vec(tet)], [:rep,:count])

c = vcat([fill(i,6) for i in 1:6]...)
cc = vec(control)
t = vcat([fill(i,6) for i in 7:12]...
tc = vec(tet)
contdf = DataFrame([c,cc],[:rep,:count])
tdf = DataFrame([t,tc],[:rep,:count])


@model function conjugate(x,y)

    α ~ Normal(500,500)
    β ~ Normal(0,500)
    σ₁ ~ Exponential(100)
    σ₂ ~ Exponential(100)

    control ~ filldist(truncated(Normal(α,σ₁),1,Inf),6)
    tet ~ filldist(truncated(Normal( α + β,σ₂),1,Inf),6)

    for i in 1:6
        x[:,i] .~ Poisson(control[i])
        y[:,i] .~ Poisson(tet[i])
    end
end
conjugate_chain = sample(conjugate(control,tet), NUTS(),MCMCThreads(),1000,4)
D = Dict()
D["chain"] = conjugate_chain
save("conjugate_chain.jld2",D)
StatsPlots.density(vcat(conjugate_chain[:β]...))

(1600 * 1200) / (170^2) # number of pixels divided by leng of scale bar
66.44 #*100 μm²
# so divide effects by 6644

