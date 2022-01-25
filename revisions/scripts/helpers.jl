function shannon(a, n)
    a .= [isnan(x) ? 0.0 : x for x in a]
    a = a[a .>0.0]
    A = sum(a)
    H = -sum([(a[i]/A) * log(a[i]/A) for i in 1:n])
    return isnan(H) ? 0 : H
end
