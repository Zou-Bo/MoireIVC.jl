using LinearAlgebra

function dm2H(dm)
    H = zeros(ComplexF64, 2,2)
    H[1,1] = -a0 * dm[1,1] + 0.5gap
    H[2,2] = -a1 * dm[2,2] - 0.5gap
    H[1,2] = -b * dm[1,2]
    H[2,1] = conj(H[1,2])
    return H
end

function energy(dm)
    H = dm2H(dm)
    E = 0.0
    E +=  real(dm[1,1] * (0.5H[1,1]+0.25gap))
    E +=  real(dm[2,2] * (0.5H[2,2]-0.25gap))
    E +=  real(dm[2,1] * 0.5H[1,2])
    E +=  real(dm[1,2] * 0.5H[2,1])
    return E
end

function sc(dm = dm0)
    dm_new = similar(dm)
    for i in 1:100
        H = dm2H(dm)
        evals, evecs = eigen(H)
        dm_new .= (evecs[:,1] * evecs[:,1]')
        dm .+= (dm_new .- dm) .* 0.5
    end
    println("energy = ", energy(dm))
    return dm
end

dm0 = [1.0 0.0; 0.0 0.0];
dm1 = [0.0 0.0; 0.0 1.0];
dm_mix = 0.5*ones(ComplexF64, 2,2);

a0 = 2.0
a1 = 1.0
b = 2.0
gap = 0.4
energy(dm0)
energy(dm1)
energy(dm_mix)

sc(copy(dm_mix))


