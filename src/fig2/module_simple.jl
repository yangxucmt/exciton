module exciton

using SpecialFunctions
using LinearAlgebra
using Plots, LaTeXStrings
using Struve
using KrylovKit
using SparseArrays
using Arpack  
using QuadGK
using ForwardDiff
using FFTW
using Measures
using TensorOperations

export GrapheneLattice, InteractionMatrix, Lattice2D, GrapheneBSE, add_bse_kernel, BlochStates, ham_index, add_wannier, envelope_real, Rshift_calculator, jxop, jyop

function ham_index(xindex, yindex; xlength, ylength)
    return xindex + (yindex - 1) * xlength
end

struct Lattice2D
    BravaisB::Vector{Float64}  # e.g., 2×2 matrix with R1, R2 as columns
    R1::Vector{Float64}
    R2::Vector{Float64}
    b1::Vector{Float64}
    b2::Vector{Float64}
end

function GrapheneLattice()
    b1 = 2π * [1, -1 / √3]
    b2 = 2π * [0, 2 / √3]
    R1 = [1, 0]
    R2 = [1 / 2, sqrt(3) / 2]
    BravaisB = (R1 - 2 * R2) / 3
    return Lattice2D(BravaisB, R1, R2, b1, b2)
end

struct InteractionMatrix
    lattice::Lattice2D
    Nsample::Int
    VAA::Array{ComplexF64,2}  # (Nsample*Nsample, Nsample*Nsample)
    VAB::Array{ComplexF64,2}  # (Nsample*Nsample, Nsample*Nsample)
end

function InteractionMatrix(lattice::Lattice2D, Nsample::Int; lambda=1, r0)
    V = lambda * Vint_Constructor(lattice, Nsample; r0)  # use lattice info
    return InteractionMatrix(lattice, Nsample, V[1, :, :], V[2, :, :])
end

function Vint_Constructor(lattice::Lattice2D, Nsample::Int; r0)
    R1 = lattice.R1
    R2 = lattice.R2
    b1 = lattice.b1
    b2 = lattice.b2
    d1 = lattice.BravaisB

    cutoff = floor(Nsample * sqrt(3) / 4) - 1

    screened_r_mat = zeros(ComplexF64, 2, Nsample, Nsample)
    for x = 1:Nsample
        for y = 1:Nsample
            screened_r_mat[1, x, y] = screened_coulomb_R(x, y, :AA; cutoff, Lx=Nsample, Ly=Nsample, R1, R2, d1, r0)
            screened_r_mat[2, x, y] = screened_coulomb_R(x, y, :AB; cutoff, Lx=Nsample, Ly=Nsample, R1, R2, d1, r0)
        end
    end

    screened_q_mat = zeros(ComplexF64, 2, Nsample, Nsample)

    phase1 = zeros(ComplexF64, Nsample, Nsample, Nsample, Nsample)
    for x = 1:Nsample
        for y = 1:Nsample
            for kx = 1:Nsample
                for ky = 1:Nsample
                    xvec = x * R1 + y * R2
                    kvec = (kx * b1 + ky * b2) / Nsample
                    phase1[x, y, kx, ky] = exp(-im * xvec' * kvec)
                end
            end
        end
    end

    @tensor begin
        screened_q_mat[1, c, d] = phase1[a, b, c, d] * screened_r_mat[1, a, b]
    end

    phase2 = zeros(ComplexF64, Nsample, Nsample, Nsample, Nsample)
    for x = 1:Nsample
        for y = 1:Nsample
            for kx = 1:Nsample
                for ky = 1:Nsample
                    xvec = x * R1 + y * R2
                    kvec = (kx * b1 + ky * b2) / Nsample
                    phase2[x, y, kx, ky] = exp(-im * xvec' * kvec)
                end
            end
        end
    end

    @tensor begin
        screened_q_mat[2, c, d] = phase2[a, b, c, d] * screened_r_mat[2, a, b]
    end

    return screened_q_mat
end

function screened_coulomb_R(rx_raw::Int, ry_raw::Int, character; cutoff, r0, Lx, Ly, R1, R2, d1) # We only compute VAA and VAB.
    function periodic_cut_discrete(x::Integer, L::Integer)
        halfL = L ÷ 2
        y = mod(x + halfL, L) - halfL
        return y
    end
    rx = periodic_cut_discrete(rx_raw, Lx)
    ry = periodic_cut_discrete(ry_raw, Ly)

    direct_vec = rx * R1 + ry * R2
    R = 0

    if character == :AA
        # Distance between A sublattices
        if rx == 0 && ry == 0
            R = 1   # on-site AA
        else
            R = norm(direct_vec)
        end
    elseif character == :AB
        # Distance from A to B sublattice
        R = norm(direct_vec + d1)
        if R < 0.01
            R = 1
        end
    else
        error("Invalid character. Use :AA or :AB.")
    end

    # Apply cutoff 
    if R > cutoff + 0.01
        return 0.0
    end

    return (struveh(0, R / r0) - bessely0(R / r0))
    # pi*aB/r0
end

function HF_coulomb(kvec, kpvec; VAA, VAB, A=[0.0, 0.0], lattice, blochstates, Nsample)
    ck = blochstates.conduction[kvec[1], kvec[2], :]
    ckp = blochstates.conduction[kpvec[1], kpvec[2], :]
    vk = blochstates.valence[kvec[1], kvec[2], :]
    vkp = blochstates.valence[kpvec[1], kpvec[2], :]

    qx, qy = mod1.(kpvec - kvec, Nsample)
    qxminus, qyminus = mod1.(kvec - kpvec, Nsample)

    Amat = [1 0; 0 0]
    Bmat = [0 0; 0 1]
    D = 0
    D += (ck' * Amat * ckp) * (vkp' * Amat * vk) * VAA[qx, qy] # :AA
    D += (ck' * Bmat * ckp) * (vkp' * Bmat * vk) * VAA[qx, qy]
    D += (ck' * Amat * ckp) * (vkp' * Bmat * vk) * VAB[qx, qy] 
    D += (ck' * Bmat * ckp) * (vkp' * Amat * vk) * VAB[qxminus, qyminus]  # :AB

    return D
end

struct BlochStates
    valence::Array{ComplexF64,3}    # (Nsample, Nsample, Nvec)
    conduction::Array{ComplexF64,3} # (Nsample, Nsample, Nvec)
    ecv::Array{Float64,2}
end

struct WannierStates
    valence::Array{ComplexF64,3}
    conduction::Array{ComplexF64,3}
end

struct GrapheneBSE
    # Lattice info
    lattice::Lattice2D

    kappa::Vector{Float64}  
    hopping::Vector{Float64}
    gap::ComplexF64
    Nsample::Int

    # Computed once
    Bloch::BlochStates
    Wannier::Union{Nothing,WannierStates}
    BSEKernel::Union{Nothing,Array{ComplexF64,2}}
end

function GrapheneBSE(lattice::Lattice2D, kappa::Vector{Float64}, hopping::Vector{Float64},
    gap, Nsample::Int)
    Bloch = compute_bloch(lattice, kappa, hopping, gap, Nsample)
    GrapheneBSE(lattice, kappa, hopping, gap, Nsample, Bloch, nothing, nothing)
end

function compute_bloch(lattice, kappa, hopping, gap, Nsample)
    b1 = lattice.b1
    b2 = lattice.b2
    R1, R2 = lattice.R1, lattice.R2
    t1 = hopping[1]
    t2 = hopping[2]
    t3 = hopping[3]
    Bloch_conduction = zeros(ComplexF64, Nsample, Nsample, 2)
    Bloch_valence = zeros(ComplexF64, Nsample, Nsample, 2)
    E_cv = zeros(Float64, Nsample, Nsample)

    for i = 1:Nsample
        for j = 1:Nsample
            k = (i * b1 + j * b2) / Nsample
            eigensys, d = eigensystem(k; t1, t2, t3, delta=gap, A=kappa, R1, R2, lattice)
            ck, vk = eachcol(eigensys)
            Bloch_conduction[i, j, :] = ck
            Bloch_valence[i, j, :] = vk
            E_cv[i, j] = 2 * d
        end
    end
    return BlochStates(Bloch_valence, Bloch_conduction, E_cv)
end

function add_bse_kernel(sys::GrapheneBSE, VInt::InteractionMatrix)
    BSEKernel = compute_bse_kernel(sys, VInt)
    return GrapheneBSE(sys.lattice,
        sys.kappa, sys.hopping, sys.gap, sys.Nsample,
        sys.Bloch, sys.Wannier, BSEKernel)
end

function compute_bse_kernel(sys::GrapheneBSE, VInt::InteractionMatrix, checkham=false)
    Nsample = sys.Nsample
    VAA = VInt.VAA
    VAB = VInt.VAB

    ham = zeros(ComplexF64, Nsample^2, Nsample^2)
    blochstates = sys.Bloch
    ecv = blochstates.ecv
    kappa = sys.kappa
    lattice = sys.lattice

    for kx = 1:Nsample, ky = 1:Nsample
        kidx = ham_index(kx, ky; xlength=Nsample, ylength=Nsample)
        ham[kidx, kidx] += ecv[kx, ky]
        for kxp = 1:Nsample, kyp = 1:Nsample
            kpidx = ham_index(kxp, kyp; xlength=Nsample, ylength=Nsample)
            ham[kidx, kpidx] += -HF_coulomb([kx, ky], [kxp, kyp]; VAA, VAB, A=kappa, lattice, blochstates, Nsample) / Nsample^2
        end
    end

    if checkham
        if !(isapprox(ham, ham'; rtol=1e-6))
            println("warning!")
        end
    end

    return (ham + ham') / 2
end

function eigensystem(k::Vector; t1, t2, t3, delta, A=[0.0, 0.0], R1, R2, lattice)
    BravaisB = lattice.BravaisB
    b1, b2 = lattice.b1, lattice.b2
    kappa = A[1] * b1 + A[2] * b2
    kA = k + kappa
    α = dot(kA, R2)
    β = dot(kA, (R2 - R1))
    Aphase = exp(im * dot(kappa, BravaisB))

    hx = -t1 - t2 * cos(α) - t3 * cos(β)
    hy = t2 * sin(α) + t3 * sin(β)
    hz = delta/2
    d = sqrt(hx^2 + hy^2 + hz^2)

    Amat = Diagonal([1, conj(Aphase)])
    Umat = sqrt((d + hz) / (2d)) * [
        1 (hx-im*hy)/(-d-hz)
        (hx+im*hy)/(d+hz) 1
    ]
    return Amat * Umat, d
end

function compute_wannier(sys::GrapheneBSE)
    Nsample = sys.Nsample
    lattice = sys.lattice
    b1, b2 = lattice.b1, lattice.b2
    R1, R2 = lattice.R1, lattice.R2
    blochstates = sys.Bloch
    Bc, Bv = blochstates.conduction, blochstates.valence
    phase = zeros(ComplexF64, Nsample, Nsample, Nsample, Nsample)
    for x = 1:Nsample
        for y = 1:Nsample
            for kx = 1:Nsample
                for ky = 1:Nsample
                    xvec = x * R1 + y * R2
                    k = (kx * b1 + ky * b2) / Nsample
                    phase[x, y, kx, ky] = exp(im * xvec' * k) / Nsample^2
                end
            end
        end
    end

    Wv = zeros(ComplexF64, Nsample, Nsample, 2)
    @tensor begin
        Wv[a, b, alpha] = phase[a, b, c, d] * Bv[c, d, alpha]
    end

    Wc = zeros(ComplexF64, Nsample, Nsample, 2)
    @tensor begin
        Wc[a, b, alpha] = phase[a, b, c, d] * Bc[c, d, alpha]
    end

    return WannierStates(Wv, Wc)
end

function add_wannier(sys::GrapheneBSE)
    Wannier = compute_wannier(sys)
    return GrapheneBSE(sys.lattice,
        sys.kappa, sys.hopping, sys.gap, sys.Nsample,
        sys.Bloch, Wannier, sys.BSEKernel)
end

function Rshift_calculator(sys::GrapheneBSE, psi, cutoff)
    function periodic_cut_discrete(x::Integer, L::Integer, cutoff)
        halfL = div(L, 2)  # integer division
        # Wrap x into [-halfL, L - halfL - 1] for odd L
        y = mod1(x, L)
        if y > halfL
            y -= L
        end
        return abs(y) <= cutoff ? y : 0
    end
    Nsample = sys.Nsample
    lattice = sys.lattice
    R1, R2 = lattice.R1, lattice.R2
    d1 = lattice.BravaisB
    basisvec = [[0, 0], d1]
    wannier_v = sys.Wannier.valence
    wannier_c = sys.Wannier.conduction

    function wannier_shift(wannier)
        wannier_R = zeros(ComplexF64, Nsample, Nsample, 2, 2) # (x,y, alpha as Bravais,vec ind)
        for alpha = 1:2
            for x = 1:Nsample
                x_cut = periodic_cut_discrete(x, Nsample, cutoff)
                for y = 1:Nsample
                    y_cut = periodic_cut_discrete(y, Nsample, cutoff)
                    wannier_R[x, y, alpha, :] = (basisvec[alpha] + x_cut * R1 + y_cut * R2) * wannier[x, y, alpha]
                end
            end
        end

        bd = [zeros(ComplexF64, 2) for _ in 1:Nsample, _ in 1:Nsample]
        for delta_x in 1:Nsample
            for delta_y in 1:Nsample
                wannier_left = zeros(ComplexF64, Nsample, Nsample, 2)
                for x = 1:Nsample
                    for y = 1:Nsample
                        wannier_left[mod1(x + delta_x, Nsample), mod1(y + delta_y, Nsample), :] = wannier[x, y, :]
                    end
                end
                wv = zeros(ComplexF64, 2)
                @tensor begin
                    wv[delta] = conj(wannier_left[a, b, alpha]) * wannier_R[a, b, alpha, delta]
                end
                bd[delta_x, delta_y] = wv
            end
        end
        return bd
    end

    bd_c = wannier_shift(wannier_c)
    bd_v = wannier_shift(wannier_v)

    Rshift = [0, 0]
    for rx_raw in -cutoff:cutoff
        rx = mod1(rx_raw, Nsample)
        for ry_raw in -cutoff:cutoff
            ry = mod1(ry_raw, Nsample)
            Rshift += (psi[rx, ry])' * psi[rx, ry] * (rx_raw * R1 + ry_raw * R2)
            for rxp_raw in -cutoff:cutoff
                rxp = mod1(rxp_raw, Nsample)
                for ryp_raw in -cutoff:cutoff
                    ryp = mod1(ryp_raw, Nsample)
                    Rshift += ((psi[rx, ry])' * psi[rxp, ryp]) * (bd_c[mod1(rxp - rx, Nsample), mod1(ryp - ry, Nsample)] - bd_v[mod1(rxp - rx, Nsample), mod1(ryp - ry, Nsample)])
                end
            end
        end
    end

    if all(abs.(imag(Rshift)) .< 1e-10)
        return real.(Rshift)
    else
        @warn "Shift vector is not real."
        print(Rshift)
        return Rshift
    end
end

function envelope_real(psik, sys::GrapheneBSE)
    Nsample = sys.Nsample
    lattice = sys.lattice
    b1, b2, R1, R2 = lattice.b1, lattice.b2, lattice.R1, lattice.R2
    psi_real = zeros(ComplexF64, Nsample, Nsample)
    for xdimr = 1:Nsample, ydimr = 1:Nsample
        real_r = xdimr * R1 + ydimr * R2
        for kx = 1:Nsample, ky = 1:Nsample
            k = (kx * b1 + ky * b2) / Nsample
            kindex = ham_index(kx, ky; xlength=Nsample, ylength=Nsample)
            psi_real[xdimr, ydimr] += exp(im * dot(real_r, k)) * psik[kindex] / Nsample
        end
    end
    return psi_real
end

function g_g(kx, ky; t1, t2, t3, delta, A=[0.0, 0.0], lattice)
    R1, R2 = lattice.R1, lattice.R2
    d1 = lattice.BravaisB
    kA = [kx, ky] .+ A
    α = dot(kA, R2)
    β = dot(kA, R2 - R1)
    Aphase = exp(im * dot(A, d1))
    hx = -t1 - t2 * cos(α) - t3 * cos(β)
    hy = t2 * sin(α) + t3 * sin(β)
    return (hx + im * hy) / Aphase
end

function jxop(kx, ky; t1, t2, t3, delta, A=[0.0, 0.0], lattice)
    df = ForwardDiff.derivative(ax -> g_g(kx, ky; A=[ax, A[2]], t1, t2, t3, delta, lattice), A[1])
    return [0 conj(df); df 0]
end

function jyop(kx, ky; t1, t2, t3, delta, A=[0.0, 0.0], lattice)
    df = ForwardDiff.derivative(ay -> g_g(kx, ky; A=[A[1], ay], t1, t2, t3, delta, lattice), A[2])
    return [0 conj(df); df 0]
end

end