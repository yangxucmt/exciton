module exciton

using SpecialFunctions
using LinearAlgebra, Base.Threads
using Plots, LaTeXStrings
using Struve
using KrylovKit
using SparseArrays
using Arpack  # For sparse eigenvalue computations
using QuadGK
using ForwardDiff
using FFTW
using Measures
using TensorOperations

export GrapheneLattice, InteractionMatrix, Lattice2D, GrapheneBSE, add_bse_kernel, BlochStates, ham_index, add_wannier, envelope_real, Rshift_calculator, jxop, jyop

function ham_index(xindex, yindex, band; nbands=2, xlength, ylength) #here band denotes (c1->v) or (c2->v), 1 has higher energy
    kindex = xindex + (yindex - 1) * xlength
    return band + (kindex - 1) * nbands
end

struct Lattice2D
    R1::Vector{Float64}
    R2::Vector{Float64}
    b1::Vector{Float64}
    b2::Vector{Float64}
end

# Factory function to initialize graphene lattice
function GrapheneLattice()
    b1 = 2π * [1, -1 / √3]
    b2 = 2π * [0, 2 / √3]
    R1 = [1, 0]
    R2 = [1 / 2, sqrt(3) / 2]
    return Lattice2D(R1, R2, b1, b2)
end

struct InteractionMatrix
    lattice::Lattice2D
    Nsample::Int
    V::Array{ComplexF64,2}  # (Nsample*Nsample, Nsample*Nsample)
end

function InteractionMatrix(lattice::Lattice2D, Nsample::Int; lambda=1, r0)
    V = lambda * Vint_Constructor(lattice, Nsample; r0)  # use lattice info
    return InteractionMatrix(lattice, Nsample, V[:, :])
end

function Vint_Constructor(lattice::Lattice2D, Nsample::Int; r0)
    R1 = lattice.R1
    R2 = lattice.R2
    b1 = lattice.b1
    b2 = lattice.b2

    cutoff = floor(Nsample * sqrt(3) / 4) - 1

    screened_r_mat = zeros(ComplexF64, Nsample, Nsample)
    for x = 1:Nsample
        for y = 1:Nsample
            screened_r_mat[x, y] = screened_coulomb_R(x, y; cutoff, Lx=Nsample, Ly=Nsample, R1, R2, r0)
        end
    end

    VR_roll = circshift(screened_r_mat, (1, 1))
    Vq_roll = fft(VR_roll)

    return circshift(Vq_roll, (-1, -1))
end

function screened_coulomb_R(rx_raw::Int, ry_raw::Int; cutoff, r0, Lx, Ly, R1, R2) # We only compute VAA and VAB.
    function periodic_cut_discrete(x::Integer, L::Integer)
        halfL = L ÷ 2
        y = mod(x + halfL, L) - halfL
        return y
    end
    rx = periodic_cut_discrete(rx_raw, Lx)
    ry = periodic_cut_discrete(ry_raw, Ly)

    direct_vec = rx * R1 + ry * R2
    R = 0

    if rx == 0 && ry == 0
        R = 1   # on-site AA
    else
        R = norm(direct_vec)
    end

    # Apply cutoff AFTER including offsets
    if R > cutoff + 0.01
        #if R > 3 + 0.01
        return 0.0
    end

    return (struveh(0, R / r0) - bessely0(R / r0))
    # pi*aB/r0
end

function HF_coulomb(kvec, kpvec, k_cond_vec; V, A=[0.0, 0.0], lattice, blochstates, Nsample)
    k_cond_ind, kp_cond_ind = k_cond_vec # (1 or 2, 1 or 2) to indicate whether c1 or c2
    conductions = [blochstates.conduction1, blochstates.conduction2]

    ck = conductions[k_cond_ind][kvec[1], kvec[2], :]
    ckp = conductions[kp_cond_ind][kpvec[1], kpvec[2], :]

    vk = blochstates.valence[kvec[1], kvec[2], :]
    vkp = blochstates.valence[kpvec[1], kpvec[2], :]

    qx, qy = mod1.(kpvec - kvec, Nsample)

    D = 0
    D += (ck' * ckp) * (vkp' * vk) * V[qx, qy] #
    return D
end

struct BlochStates
    valence::Array{ComplexF64,3}    # (Nsample, Nsample, Nvec)
    conduction1::Array{ComplexF64,3} # (Nsample, Nsample, Nvec,cond_band_ind=2), c1 has higher energy than c2
    conduction2::Array{ComplexF64,3} # (Nsample, Nsample, Nvec,cond_band_ind=2), c1 has higher energy than c2
    # ecv::Array{Float64,3} # the last index signaling whether it is (c1->v) or (c2->v)
    ecvmat::Union{Nothing,Array{ComplexF64,4}} # (Nsample, Nsample, 2, 2) the last two are particle-hole basis (c1->v) and (c2->v)
end


struct WannierStates
    valence::Array{ComplexF64,3}
    conduction1::Array{ComplexF64,3}
    conduction2::Array{ComplexF64,3}
end

struct GrapheneBSE
    # Lattice info
    lattice::Lattice2D

    # User inputs
    kappa::Vector{Float64}
    Nsample::Int

    # Computed once
    Bloch::BlochStates
    Wannier::Union{Nothing,WannierStates}
    BSEKernel::Union{Nothing,Array{ComplexF64,2}}
end

function invsqrt_hermitian(S::AbstractMatrix; eps=1e-3)
    F = eigen(S)
    #=
    if any(abs.(F.values) .< eps)
        @warn "Some eigenvalues are smaller than $eps, inversion may be unstable."
    end
    =#
    return F.vectors * Diagonal(F.values .^ (-0.5)) * F.vectors'
    #=
        U, σ, V = svd(S)
        if any(abs.(σ) .< eps)
            @warn "Some eigenvalues are smaller than $eps, inversion may be unstable."
        end
        S_inv_sqrt = U * Diagonal(σ .^ (-0.5)) * U'
        return S_inv_sqrt
        =#
end

function gauge_fixing(psi, g_list)
    nbands = length(psi)

    # Construct φ_i = sum_k ψ_k (ψ_k' * g_i)
    phi = Vector{typeof(psi[1])}(undef, nbands)
    for i in 1:nbands
        φ = zero(psi[1])
        for k in 1:nbands
            φ += psi[k] * (psi[k]' * g_list[i])
        end
        phi[i] = φ
    end

    # Build overlap matrix S_ij = φ_i' * φ_j
    smat = Matrix{ComplexF64}(undef, nbands, nbands)
    for i in 1:nbands
        for j in 1:nbands
            smat[i, j] = phi[i]' * phi[j]
        end
    end

    # Inverse square root of S
    sinvmat = nbands == 1 ? 1 / sqrt(real(smat[1, 1])) : invsqrt_hermitian(smat)

    phi_renormalized = Vector{typeof(phi[1])}(undef, nbands)
    for n in 1:nbands
        φ̃ = zero(phi[1])
        for m in 1:nbands
            φ̃ += phi[m] * sinvmat[m, n]
        end
        phi_renormalized[n] = φ̃
    end

    umat_mn = zeros(ComplexF64, nbands, nbands)
    for mind = 1:nbands, nind = 1:nbands, lind = 1:nbands
        umat_mn[mind, nind] += (psi[mind]' * g_list[lind]) * sinvmat[lind, nind]
    end

    # the definition of umat is: \tilde{\phi}_n=\sum\limits_m umat_mn * \psi_m

    return umat_mn
end

function projection_gauge_fixing(Bloch, Nsample; sz, lattice)
    #gc1 = [1, 0.0, 0]
    gc1 = [1 + 0.2 * im, 0.2 - 0.1 * im, 0.3]
    gc1 = gc1 ./ norm(gc1)
    #gc2 = [0.0, 1, im]
    gc2 = [0.2 - 0.1 * im, 1 - 0.3 * im, im + 0.2]
    gc2 = gc2 ./ norm(gc2)

    b1, b2 = lattice.b1, lattice.b2

    wmat_c1, wmat_c2 = Bloch.conduction1, Bloch.conduction2
    c1list = 0 * wmat_c1
    c2list = 0 * wmat_c2
    ecvmat = Bloch.ecvmat
    for xi = 1:Nsample, yi = 1:Nsample
        k = (xi * b1 + yi * b2) / Nsample

        umat_mn = gauge_fixing([wmat_c1[xi, yi, :], wmat_c2[xi, yi, :]], [gc1, gc2])
        phi = [wmat_c1[xi, yi, :], wmat_c2[xi, yi, :]]
        phi_new = [0 * wmat_c1[1, 1, :], 0 * wmat_c1[1, 1, :]]

        for n in 1:2
            for m in 1:2
                phi_new[n] += phi[m] * umat_mn[m, n]
            end
        end

        c1list[xi, yi, :], c2list[xi, yi, :] = phi_new

        new_ecv = umat_mn' * ecvmat[xi, yi, :, :] * umat_mn
        ecvmat[xi, yi, :, :] = new_ecv
    end

    #gv = [0, 1, -im]

    wmat_v = Bloch.valence
    vlist = 0 * wmat_v

    gv = [0.4 - 0.1im, -0.3 + 0.1im, -0.4 + 0.15im]
    gv = gv ./ norm(gv)

    #  gv = wmat_v[Nsample, Nsample, :]

    for xi = 1:Nsample, yi = 1:Nsample
        #gv = [0, sqrt(3)/2, im*sin((xi-yi)*2pi/Nsample)]
        umat_mn = gauge_fixing([wmat_v[xi, yi, :]], [gv])
        vvec = wmat_v[xi, yi, :] * umat_mn[1]
        vlist[xi, yi, :] = vvec
    end

    println("gauge fixing successful!")

    return BlochStates(vlist, c1list, c2list, ecvmat)
end

# Outer constructor (factory)
function GrapheneBSE(lattice::Lattice2D, kappa::Vector{Float64}, Nsample::Int; sz, epsilonyy)
    Bloch = compute_bloch(lattice, kappa, Nsample; sz, epsilonyy)
    Bloch_fixed = projection_gauge_fixing(Bloch, Nsample; sz, lattice)
    GrapheneBSE(lattice, kappa, Nsample, Bloch_fixed, nothing, nothing)
end

function TMD_hnn(k; epsilonyy, sz)
    kx, ky = k
    e1 = 1.046
    e2 = 2.104
    t0 = -0.184
    t1 = 0.401
    t2 = 0.507
    t11 = 0.218
    t12 = 0.338
    t22 = 0.057
    lambda_so = 0.073
    alphaval = kx / 2
    betaval = sqrt(3) * ky / 2
    c3_theta = 2pi / 3
    c3_mat = [1 0 0; 0 cos(c3_theta) -sin(c3_theta); 0 sin(c3_theta) cos(c3_theta)]

    sigmav = [1 0 0; 0 -1 0; 0 0 1]

    Thopping = [t0 t1 t2; -t1 t11 t12; t2 -t12 t22]

    cT = c3_mat * Thopping * inv(c3_mat)
    sigmaT = sigmav * Thopping * inv(sigmav)
    csigmaT = c3_mat * sigmaT * inv(c3_mat)
    c2T = c3_mat * cT * inv(c3_mat)
    c2sigmaT = c3_mat * csigmaT * inv(c3_mat)
    ham_expr = exp(im * 2 * alphaval) * Thopping * (1 + 0) + exp(-im * 2 * alphaval) * sigmaT * (1 + 0)
    ham_expr += (exp(im * (betaval - alphaval)) * cT + exp(-im * (betaval - alphaval)) * csigmaT) * (1 + epsilonyy)
    ham_expr += (exp(im * (-betaval - alphaval)) * c2T + exp(im * (betaval + alphaval)) * c2sigmaT) * (1 + epsilonyy)
    ham_expr += [e1 0 0; 0 e2 0; 0 0 e2]
    ham_expr += [0 0 0; 0 0 2im; 0 -2im 0] * lambda_so * sz / 2
    return ham_expr
end

function compute_bloch(lattice, kappa, Nsample; sz, epsilonyy)
    b1 = lattice.b1
    b2 = lattice.b2
    R1, R2 = lattice.R1, lattice.R2
    Bloch_conduction1 = zeros(ComplexF64, Nsample, Nsample, 3)
    Bloch_conduction2 = zeros(ComplexF64, Nsample, Nsample, 3)
    Bloch_valence = zeros(ComplexF64, Nsample, Nsample, 3)
    E_cv = zeros(ComplexF64, Nsample, Nsample, 2, 2)

    for i = 1:Nsample
        for j = 1:Nsample
            k = (i * b1 + j * b2) / Nsample
            mat = TMD_hnn(k; sz, epsilonyy)
            eige, eigv = eigen(mat)
            vk, c2k, c1k = eachcol(eigv)
            Bloch_conduction1[i, j, :] = c1k
            Bloch_conduction2[i, j, :] = c2k
            Bloch_valence[i, j, :] = vk
            E_cv[i, j, 1, 1] = (eige[3] - eige[1])
            E_cv[i, j, 2, 2] = (eige[2] - eige[1])
        end
    end
    return BlochStates(Bloch_valence, Bloch_conduction1, Bloch_conduction2, E_cv)
end

function add_bse_kernel(sys::GrapheneBSE, VInt::InteractionMatrix)
    BSEKernel = compute_bse_kernel_fast_parallel(sys, VInt)
    return GrapheneBSE(sys.lattice,
        sys.kappa, sys.Nsample,
        sys.Bloch, sys.Wannier, BSEKernel)
end

function compute_bse_kernel(sys::GrapheneBSE, VInt::InteractionMatrix, checkham=true)
    Nsample = sys.Nsample
    V = VInt.V

    ham = zeros(ComplexF64, Nsample^2 * 2, Nsample^2 * 2)
    blochstates = sys.Bloch
    #ecv = blochstates.ecv
    ecvmat = blochstates.ecvmat
    kappa = sys.kappa
    lattice = sys.lattice

    for kx = 1:Nsample, ky = 1:Nsample, k_cond_ind = 1:2
        kidx = ham_index(kx, ky, k_cond_ind; xlength=Nsample, ylength=Nsample)
        for kxp = 1:Nsample, kyp = 1:Nsample, kp_cond_ind = 1:2
            kpidx = ham_index(kxp, kyp, kp_cond_ind; xlength=Nsample, ylength=Nsample)
            ham[kidx, kpidx] += -HF_coulomb([kx, ky], [kxp, kyp], [k_cond_ind, kp_cond_ind]; V, A=kappa, lattice, blochstates, Nsample) / Nsample^2
        end
    end

    for kx = 1:Nsample, ky = 1:Nsample
        for k_cond_ind = 1:2
            for kp_cond_ind = 1:2
                kidx = ham_index(kx, ky, k_cond_ind; xlength=Nsample, ylength=Nsample)
                kpidx = ham_index(kx, ky, kp_cond_ind; xlength=Nsample, ylength=Nsample)
                ham[kidx, kpidx] += ecvmat[kx, ky, k_cond_ind, kp_cond_ind]
            end
        end
    end

    if checkham
        if !(isapprox(ham, ham'; rtol=1e-6))
            println("warning!")
        end
    end

    return (ham + ham') / 2
end

function compute_bse_kernel_fast(sys::GrapheneBSE, VInt::InteractionMatrix; checkham=true)
    Nsample = sys.Nsample
    V = VInt.V
    kappa = sys.kappa
    lattice = sys.lattice
    blochstates = sys.Bloch
    ecvmat = blochstates.ecvmat

    # Precompute all Hamiltonian indices
    ham_indices = Array{Int}(undef, Nsample, Nsample, 2)
    for kx in 1:Nsample, ky in 1:Nsample, k_cond_ind in 1:2
        ham_indices[kx, ky, k_cond_ind] = ham_index(kx, ky, k_cond_ind; xlength=Nsample, ylength=Nsample)
    end

    ham = zeros(ComplexF64, Nsample^2 * 2, Nsample^2 * 2)

    # Coulomb part
    for kx in 1:Nsample, ky in 1:Nsample, k_cond_ind in 1:2
        kidx = ham_indices[kx, ky, k_cond_ind]
        for kxp in 1:Nsample, kyp in 1:Nsample, kp_cond_ind in 1:2
            kpidx = ham_indices[kxp, kyp, kp_cond_ind]
            ham[kidx, kpidx] += -HF_coulomb([kx, ky], [kxp, kyp], [k_cond_ind, kp_cond_ind];
                V, A=kappa, lattice, blochstates, Nsample) / Nsample^2
        end
    end

    # Add ecvmat contribution (same k)
    for kx in 1:Nsample, ky in 1:Nsample, k_cond_ind in 1:2, kp_cond_ind in 1:2
        kidx = ham_indices[kx, ky, k_cond_ind]
        kpidx = ham_indices[kx, ky, kp_cond_ind]
        ham[kidx, kpidx] += ecvmat[kx, ky, k_cond_ind, kp_cond_ind]
    end

    if checkham && !(isapprox(ham, ham'; rtol=1e-6))
        println("warning! Hamiltonian is not Hermitian")
    end

    return (ham + ham') / 2
end

function compute_bse_kernel_fast_parallel(sys::GrapheneBSE, VInt::InteractionMatrix; checkham=true)
    Nsample = sys.Nsample
    V = VInt.V
    kappa = sys.kappa
    lattice = sys.lattice
    blochstates = sys.Bloch
    ecvmat = blochstates.ecvmat

    # Precompute all Hamiltonian indices (This is fast and can remain sequential)
    ham_indices = Array{Int}(undef, Nsample, Nsample, 2)
    for kx in 1:Nsample, ky in 1:Nsample, k_cond_ind in 1:2
        ham_indices[kx, ky, k_cond_ind] = ham_index(kx, ky, k_cond_ind; xlength=Nsample, ylength=Nsample)
    end

    # Initialize the Hamiltonian matrix
    # Using a thread-local accumulation strategy is safer for complex matrix writes, 
    # but here, since indices kidx and kpidx are unique for each (kx, ky, k_cond_ind, kxp, kyp, kp_cond_ind) 
    # in the Coulomb loop, a direct write is safe.
    ham = zeros(ComplexF64, Nsample^2 * 2, Nsample^2 * 2)

    # 1. Coulomb part (Highly parallelizable due to two independent k-point loops)
    # Parallelize the outermost loop for the best distribution of work.
    @threads for kx in 1:Nsample
        for ky in 1:Nsample, k_cond_ind in 1:2
            kidx = ham_indices[kx, ky, k_cond_ind]

            for kxp in 1:Nsample, kyp in 1:Nsample, kp_cond_ind in 1:2
                kpidx = ham_indices[kxp, kyp, kp_cond_ind]

                # Direct assignment (+=) is safe here because the indices (kidx, kpidx)
                # are unique for each iteration of the parallelized loop.
                ham[kidx, kpidx] += -HF_coulomb([kx, ky], [kxp, kyp], [k_cond_ind, kp_cond_ind];
                    V, A=kappa, lattice, blochstates, Nsample) / Nsample^2
            end
        end
    end

    # 2. Add ecvmat contribution (Less expensive, but still parallelized)
    # The loops kxp, kyp, kp_cond_ind are now replaced by an inner loop structure.
    # Parallelize the outermost loop (kx).
    @threads for kx in 1:Nsample
        for ky in 1:Nsample
            for k_cond_ind in 1:2, kp_cond_ind in 1:2
                kidx = ham_indices[kx, ky, k_cond_ind]
                kpidx = ham_indices[kx, ky, kp_cond_ind]
                # Direct assignment (+=) is safe here for the same reason.
                ham[kidx, kpidx] += ecvmat[kx, ky, k_cond_ind, kp_cond_ind]
            end
        end
    end

    if checkham && !(isapprox(ham, ham'; rtol=1e-6))
        println("warning! Hamiltonian is not Hermitian")
    end

    # Ensure the returned matrix is perfectly Hermitian
    return (ham + ham') / 2
end


# ------------------------------------------------------------------------------------------ #
# ------------------------------- Below are for gauge fixing ------------------------------- #
# ------------------------------------------------------------------------------------------ #


function ifft_routine(A)
    A_roll = circshift(A, (1, 1))

    A_ifft = ifft(A_roll)
    return circshift(A_ifft, (-1, -1))
end

function compute_wannier_fft(sys::GrapheneBSE)
    Nsample = sys.Nsample
    lattice = sys.lattice
    b1, b2 = lattice.b1, lattice.b2
    R1, R2 = lattice.R1, lattice.R2
    blochstates = sys.Bloch
    Bc1, Bc2, Bv = blochstates.conduction1, blochstates.conduction2, blochstates.valence

    Wv = zeros(ComplexF64, Nsample, Nsample, 3)
    Wc1 = zeros(ComplexF64, Nsample, Nsample, 3)
    Wc2 = zeros(ComplexF64, Nsample, Nsample, 3)

    for alpha = 1:3
        Bmat = Bv[:, :, alpha]
        Wv[:, :, alpha] = ifft_routine(Bmat)
    end

    for alpha = 1:3
        Bmat = Bc1[:, :, alpha]
        Wc1[:, :, alpha] = ifft_routine(Bmat)
    end

    for alpha = 1:3
        Bmat = Bc2[:, :, alpha]
        Wc2[:, :, alpha] = ifft_routine(Bmat)
    end

    return WannierStates(Wv, Wc1, Wc2)
end

# --- Methods to "upgrade" the struct ---
function add_wannier(sys::GrapheneBSE)
    Wannier = compute_wannier_fft(sys)
    return GrapheneBSE(sys.lattice,
        sys.kappa, sys.Nsample,
        sys.Bloch, Wannier, sys.BSEKernel)
end

function Rshift_calculator(sys::GrapheneBSE, psi, cutoff; strainyy=0)
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
    R1, R2 = lattice.R1, [1, (1 + strainyy)] .* lattice.R2
    wannier_v = sys.Wannier.valence
    wannier_c1 = sys.Wannier.conduction1
    wannier_c2 = sys.Wannier.conduction2

    psi1, psi2 = psi

    function wannier_shift(wannier_left, wannier_right)
        wannier_R = zeros(ComplexF64, Nsample, Nsample, 3, 2) # (x,y, vec ind)
        for alpha = 1:3
            for x = 1:Nsample
                x_cut = periodic_cut_discrete(x, Nsample, cutoff)
                for y = 1:Nsample
                    y_cut = periodic_cut_discrete(y, Nsample, cutoff)
                    wannier_R[x, y, alpha, :] = (x_cut * R1 + y_cut * R2) * wannier_right[x, y, alpha]
                end
            end
        end

        bd = [zeros(ComplexF64, 2) for _ in 1:Nsample, _ in 1:Nsample]
        for delta_x in 1:Nsample
            for delta_y in 1:Nsample
                wannier_left_shifted = zeros(ComplexF64, Nsample, Nsample, 3)
                for x = 1:Nsample
                    for y = 1:Nsample
                        wannier_left_shifted[mod1(x - delta_x, Nsample), mod1(y - delta_y, Nsample), :] = wannier_left[x, y, :]
                    end
                end
                wv = zeros(ComplexF64, 2)
                @tensor begin
                    wv[delta] = conj(wannier_left_shifted[a, b, alpha]) * wannier_R[a, b, alpha, delta]
                end
                bd[delta_x, delta_y] = wv
            end
        end
        return bd
    end

    bd_c11 = wannier_shift(wannier_c1, wannier_c1)
    bd_c12 = wannier_shift(wannier_c1, wannier_c2)
    bd_c22 = wannier_shift(wannier_c2, wannier_c2)
    bd_c21 = wannier_shift(wannier_c2, wannier_c1)

    bd_v = wannier_shift(wannier_v, wannier_v)

    Rshift = [0, 0]
    Rshift_psi = [0, 0]
    for rx_raw in -cutoff:cutoff
        rx = mod1(rx_raw, Nsample)
        for ry_raw in -cutoff:cutoff
            ry = mod1(ry_raw, Nsample)
            Rshift += (psi1[rx, ry])' * psi1[rx, ry] * (rx_raw * R1 + ry_raw * R2)
            Rshift += (psi2[rx, ry])' * psi2[rx, ry] * (rx_raw * R1 + ry_raw * R2)

            Rshift_psi += (psi1[rx, ry])' * psi1[rx, ry] * (rx_raw * R1 + ry_raw * R2)
            Rshift_psi += (psi2[rx, ry])' * psi2[rx, ry] * (rx_raw * R1 + ry_raw * R2)

            for rxp_raw in -cutoff:cutoff
                rxp = mod1(rxp_raw, Nsample)
                for ryp_raw in -cutoff:cutoff
                    ryp = mod1(ryp_raw, Nsample)
                    Rshift += ((psi1[rx, ry])' * psi1[rxp, ryp]) * (bd_c11[mod1(rxp - rx, Nsample), mod1(ryp - ry, Nsample)] - bd_v[mod1(rxp - rx, Nsample), mod1(ryp - ry, Nsample)])
                    Rshift += ((psi2[rx, ry])' * psi2[rxp, ryp]) * (bd_c22[mod1(rxp - rx, Nsample), mod1(ryp - ry, Nsample)] - bd_v[mod1(rxp - rx, Nsample), mod1(ryp - ry, Nsample)])
                    Rshift += ((psi1[rx, ry])' * psi2[rxp, ryp]) * (bd_c12[mod1(rxp - rx, Nsample), mod1(ryp - ry, Nsample)])
                    Rshift += ((psi2[rx, ry])' * psi1[rxp, ryp]) * (bd_c21[mod1(rxp - rx, Nsample), mod1(ryp - ry, Nsample)])
                end
            end
        end
    end


    if all(abs.(imag(Rshift)) .< 1e-8)
        println(real.(Rshift))
        return real.(Rshift)
    else
        @warn "Shift vector is not real."
        print(Rshift)
        return Rshift
    end
end

function envelope_real_fft(psik, sys::GrapheneBSE)
    Nsample = sys.Nsample
    lattice = sys.lattice
    b1, b2, R1, R2 = lattice.b1, lattice.b2, lattice.R1, lattice.R2

    psik1_fft = zeros(ComplexF64, Nsample, Nsample)
    psik2_fft = zeros(ComplexF64, Nsample, Nsample)

    for kx = 1:Nsample, ky = 1:Nsample
        kindex = ham_index(kx, ky, 1; xlength=Nsample, ylength=Nsample)
        psik1_fft[kx, ky] = psik[kindex]
    end

    for kx = 1:Nsample, ky = 1:Nsample
        kindex = ham_index(kx, ky, 2; xlength=Nsample, ylength=Nsample)
        psik2_fft[kx, ky] = psik[kindex]
    end

    return [ifft_routine(psik1_fft) * Nsample, ifft_routine(psik2_fft) * Nsample]
end

function wigner_seitz(rvec, lattice, Nsample; tol=1e-6)
    R1, R2 = lattice.R1, lattice.R2
    candidates = [rvec, rvec .- Nsample * R1, rvec .- Nsample * R2, rvec .- Nsample * R1 .- Nsample * R2]
    norms = [norm(c) for c in candidates]
    minval = minimum(norms)
    idxs = findall(x -> abs(x - minval) < tol, norms)

    if length(idxs) == 1
        return candidates[idxs[1]]
    else
        # --- tie-breaking rule based on C3 symmetry axes ---
        dirs = [Nsample * R1, Nsample * R2 - Nsample * R1, -Nsample * R2]

        scores = [maximum(dot(c, d) for d in dirs) for c in candidates[idxs]]
        return candidates[idxs[argmax(scores)]]
    end
end


function Gk_subroutine(sub_M_blist; wb, nbands, blist, rlist)
    Rmat = zeros(ComplexF64, nbands, nbands)
    Tmat = zeros(ComplexF64, nbands, nbands)
    for bind = 1:3
        Mmat = sub_M_blist[bind]
        for mind = 1:nbands, nind = 1:nbands
            Rmat[mind, nind] += Mmat[mind, nind] * conj(Mmat[nind, nind])
            qn = angle(Mmat[nind, nind]) + blist[bind]' * rlist[nind]
            Tmat[mind, nind] += Mmat[mind, nind] / Mmat[nind, nind] * qn
        end
    end

    Gk = 4 * wb * ((Rmat - Rmat') / 2 - (Tmat + Tmat') / (2im))
    return Gk
    # should return a matrix
end

function Mb_prep(Umn_mat; M0_blist, Nsample)
    Mmn_mat_b1, Mmn_mat_b2, Mmn_mat_b3 = M0_blist
    updated_Mmn_mat_b1 = 0 * Mmn_mat_b1
    updated_Mmn_mat_b2 = 0 * Mmn_mat_b2
    updated_Mmn_mat_b3 = 0 * Mmn_mat_b3

    for xi = 1:Nsample, yi = 1:Nsample
        updated_Mmn_mat_b1[xi, yi, :, :] = Umn_mat[xi, yi, :, :]' * Mmn_mat_b1[xi, yi, :, :] * Umn_mat[mod1(xi + 1, Nsample), yi, :, :]
        updated_Mmn_mat_b2[xi, yi, :, :] = Umn_mat[xi, yi, :, :]' * Mmn_mat_b2[xi, yi, :, :] * Umn_mat[xi, mod1(yi + 1, Nsample), :, :]
        updated_Mmn_mat_b3[xi, yi, :, :] = Umn_mat[xi, yi, :, :]' * Mmn_mat_b3[xi, yi, :, :] * Umn_mat[mod1(xi - 1, Nsample), mod1(yi - 1, Nsample), :, :]
    end

    return [updated_Mmn_mat_b1, updated_Mmn_mat_b2, updated_Mmn_mat_b3]
end

function compute_M_matrix(bloch_states::Vector{Array{ComplexF64,3}}, shift::Tuple{Int,Int})
    Nsample = size(bloch_states[1], 1)
    nbands = length(bloch_states)  # assume array of vectors per k
    Mmn_mat = zeros(ComplexF64, Nsample, Nsample, nbands, nbands)

    dx, dy = shift
    for xi = 1:Nsample, yi = 1:Nsample
        xi2 = mod1(xi + dx, Nsample)
        yi2 = mod1(yi + dy, Nsample)
        for m = 1:nbands, n = 1:nbands
            Mmn_mat[xi, yi, m, n] = bloch_states[m][xi, yi, :]' * bloch_states[n][xi2, yi2, :]
        end
    end
    return Mmn_mat
end

function omega_spreading(Mblist; wb, Nsample, blist, rlist)
    nbands = size(Mblist[1])[3]
    M_b1, M_b2, M_b3 = Mblist

    omega_value = 0
    for xi = 1:Nsample, yi = 1:Nsample, bandind = 1:nbands, bi_ind = 1:3
        omega_value += wb * (1 - abs(Mblist[bi_ind][xi, yi, bandind, bandind])^2) / Nsample^2
        qn = angle(Mblist[bi_ind][xi, yi, bandind, bandind]) + blist[bi_ind]' * rlist[bandind]
        omega_value += wb * qn^2 / Nsample^2
    end

    return omega_value
end

function omega_invariant_spreading(Mblist; wb, Nsample)
    M_b1, M_b2, M_b3 = Mblist
    nbands = size(Mblist[1])[3]
    omega_value = 0
    for xi = 1:Nsample, yi = 1:Nsample, bi_ind = 1:3
        omega_value += wb * nbands / Nsample^2
        for bandind_x = 1:nbands, bandind_y = 1:nbands
            omega_value += -wb * abs(Mblist[bi_ind][xi, yi, bandind_x, bandind_y])^2 / Nsample^2
        end
    end
    return omega_value
end

function one_descent_step_singleband(Umn_mat; M0_blist, wb, alphaval=0.3, varied_step=false, Nsample, blist, rlist, verbose=true)
    updated_Mblist = Mb_prep(Umn_mat; M0_blist, Nsample)
    if verbose
        println("Total spreading: ", omega_spreading(updated_Mblist; wb, Nsample, blist, rlist), "; Invariant part: ", omega_invariant_spreading(updated_Mblist; wb, Nsample))
    end
    M_b1, M_b2, M_b3 = updated_Mblist
    updated_Umn_mat = similar(Umn_mat)

    # Step 1: update Umn_mat
    for xi = 1:Nsample, yi = 1:Nsample
        Gk = 4 * im * wb * imag(log(M_b1[xi, yi, 1, 1]) + log(M_b2[xi, yi, 1, 1]) + log(M_b3[xi, yi, 1, 1]))
        # used 0.3

        deltaWk = 0
        if varied_step
            gknorm = norm(1 / (4 * wb) * Gk)
            max_step = 0.1
            # scale down deltaWk if gradient is too large
            scale = min(1.0, max_step / gknorm)
            deltaWk = scale * (alphaval / (4 * wb) * Gk)
        else
            deltaWk = alphaval / (4 * wb) * Gk
        end
        updated_Umn_mat[xi, yi, :, :] = Umn_mat[xi, yi, :, :] * exp(deltaWk)
    end

    # Step 2: update Mblist using the new Umn_mat

    return updated_Umn_mat
end

function one_descent_step(Umn_mat; M0_blist, wb, alphaval=0.3, nbands, Nsample, blist, rlist, verbose=true)
    updated_Mblist = Mb_prep(Umn_mat; M0_blist, Nsample)
    if verbose
        println("Total spreading: ", omega_spreading(updated_Mblist; wb, Nsample, blist, rlist), "; Invariant part: ", omega_invariant_spreading(updated_Mblist; wb, Nsample))
    end
    M_b1, M_b2, M_b3 = updated_Mblist
    updated_Umn_mat = similar(Umn_mat)

    # Step 1: update Umn_mat
    for xi = 1:Nsample, yi = 1:Nsample
        sub_M_b1 = M_b1[xi, yi, :, :]
        sub_M_b2 = M_b2[xi, yi, :, :]
        sub_M_b3 = M_b3[xi, yi, :, :]
        sub_M_blist = [sub_M_b1, sub_M_b2, sub_M_b3]

        Gk = Gk_subroutine(sub_M_blist; wb, nbands, blist, rlist)
        # used 0.3
        deltaWk = alphaval / (4 * wb) * Gk

        updated_Umn_mat[xi, yi, :, :] = Umn_mat[xi, yi, :, :] * exp(deltaWk)
    end

    # Step 2: update Mblist using the new Umn_mat

    return updated_Umn_mat
end


function exciton_shift_current_subroutine(Nsample, epsilonyy, sz, th_angle)
    px = cos(th_angle)
    py = sin(th_angle)

    lattice = GrapheneLattice()
    #VInt = InteractionMatrix(lattice, Nsample; lambda=0.8, r0=4.28)
    VInt = InteractionMatrix(lattice, Nsample; lambda=0.667, r0=4.288)

    bi_1 = lattice.b1 / Nsample
    bi_2 = lattice.b2 / Nsample
    bi_3 = -bi_1 - bi_2
    wb = 2 / (3 * norm(bi_1)^2)

    graphene_sys = GrapheneBSE(lattice, [0.0, 0], Nsample; sz, epsilonyy)
    bloch_v, bloch_c1, bloch_c2 = graphene_sys.Bloch.valence, graphene_sys.Bloch.conduction1, graphene_sys.Bloch.conduction2
    bloch_states = [bloch_v]
    M_b1 = compute_M_matrix(bloch_states, (1, 0))
    M_b2 = compute_M_matrix(bloch_states, (0, 1))
    M_b3 = compute_M_matrix(bloch_states, (-1, -1))
    blist = [bi_1, bi_2, bi_3]
    M_blist = [M_b1, M_b2, M_b3]
    nbands = length(bloch_states)

    rlist = [
        sum(-wb / Nsample^2 * blist[bind] * angle(M_blist[bind][xi, yi, band, band])
            for xi = 1:Nsample, yi = 1:Nsample, bind = 1:3)
        for band = 1:nbands
    ]

    nbands = length(bloch_states)
    U0mn_mat = zeros(ComplexF64, Nsample, Nsample, nbands, nbands) # Initialization
    for xi = 1:Nsample, yi = 1:Nsample, nband_ind = 1:nbands
        U0mn_mat[xi, yi, nband_ind, nband_ind] = 1
    end

    updated_Umn_mat = one_descent_step_singleband(U0mn_mat; M0_blist=M_blist, wb, varied_step=false, Nsample, blist, rlist, verbose=false)
    for steps = 1:200
        updated_Umn_mat = one_descent_step_singleband(updated_Umn_mat; M0_blist=M_blist, wb, varied_step=false, alphaval=0.1, Nsample, blist, rlist, verbose=false)
    end

    oldv = graphene_sys.Bloch.valence
    newv = 0 * oldv

    for xi = 1:Nsample, yi = 1:Nsample
        newv[xi, yi, :] = updated_Umn_mat[xi, yi, 1, 1] * oldv[xi, yi, :]
    end

    bloch_states = [bloch_c1, bloch_c2]
    M_b1 = compute_M_matrix(bloch_states, (1, 0))
    M_b2 = compute_M_matrix(bloch_states, (0, 1))
    M_b3 = compute_M_matrix(bloch_states, (-1, -1))

    blist = [bi_1, bi_2, bi_3]
    M_blist = [M_b1, M_b2, M_b3]
    nbands = length(bloch_states)

    rlist = [
        sum(-wb / Nsample^2 * blist[bind] * angle(M_blist[bind][xi, yi, band, band])
            for xi = 1:Nsample, yi = 1:Nsample, bind = 1:3)
        for band = 1:nbands
    ]

    nbands = length(bloch_states)
    U0mn_mat = zeros(ComplexF64, Nsample, Nsample, nbands, nbands) # Initialization
    for xi = 1:Nsample, yi = 1:Nsample, nband_ind = 1:nbands
        U0mn_mat[xi, yi, nband_ind, nband_ind] = 1
    end
    updated_Umn_mat = one_descent_step(U0mn_mat; M0_blist=M_blist, wb, nbands, Nsample, blist, rlist, verbose=false)
    for steps = 1:200
        updated_Umn_mat = one_descent_step(updated_Umn_mat; M0_blist=M_blist, wb, nbands, Nsample, blist, rlist, verbose=false)
    end

    oldc1 = graphene_sys.Bloch.conduction1
    oldc2 = graphene_sys.Bloch.conduction2
    newc1 = 0 * oldc1
    newc2 = 0 * oldc2

    for xi = 1:Nsample, yi = 1:Nsample
        newc1[xi, yi, :] = updated_Umn_mat[xi, yi, 1, 1] * oldc1[xi, yi, :] + updated_Umn_mat[xi, yi, 2, 1] * oldc2[xi, yi, :]
        newc2[xi, yi, :] = updated_Umn_mat[xi, yi, 1, 2] * oldc1[xi, yi, :] + updated_Umn_mat[xi, yi, 2, 2] * oldc2[xi, yi, :]
    end

    oldecv = graphene_sys.Bloch.ecvmat
    newecv = 0 * oldecv
    for xi = 1:Nsample, yi = 1:Nsample
        urotated = updated_Umn_mat[xi, yi, :, :]
        newecv[xi, yi, :, :] = urotated' * oldecv[xi, yi, :, :] * urotated
    end

    mlwf_bloch = BlochStates(newv, newc1, newc2, newecv)
    graphene_sys = GrapheneBSE(graphene_sys.lattice, graphene_sys.kappa, graphene_sys.Nsample, mlwf_bloch, nothing, nothing)
    graphene_sys = add_bse_kernel(graphene_sys, VInt)
    bsemat = graphene_sys.BSEKernel
    xlen = size(bsemat)[1]
    x0 = rand(xlen)
    valslist, vecslist, info = eigsolve(bsemat, x0, 10, :SR,
        krylovdim=40, tol=1e-10, maxiter=20,
        verbosity=0, ishermitian=true)
    graphene_sys = add_wannier(graphene_sys)
    psik = vecslist[1]
    psi_real = envelope_real_fft(psik, graphene_sys)
    r1 = Rshift_calculator(graphene_sys, psi_real, 30)

    psik1_fft = zeros(ComplexF64, Nsample, Nsample)
    psik2_fft = zeros(ComplexF64, Nsample, Nsample)

    for kx = 1:Nsample, ky = 1:Nsample
        kindex = ham_index(kx, ky, 1; xlength=Nsample, ylength=Nsample)
        psik1_fft[kx, ky] = psik[kindex]
    end

    for kx = 1:Nsample, ky = 1:Nsample
        kindex = ham_index(kx, ky, 2; xlength=Nsample, ylength=Nsample)
        psik2_fft[kx, ky] = psik[kindex]
    end
    gbloch = graphene_sys.Bloch
    gv, gc1, gc2 = gbloch.valence, gbloch.conduction1, gbloch.conduction2

    r_elem = 0

    for xi = 1:Nsample, yi = 1:Nsample
        k = (lattice.b1 * xi + lattice.b2 * yi) / Nsample
        vop = TMD_jx_6th(k; epsilonyy, sz) * px + TMD_jy_6th(k; epsilonyy, sz) * py

        r_elem += (gv[xi, yi, :]' * vop * gc1[xi, yi, :]) * psik1_fft[xi, yi]
        r_elem += (gv[xi, yi, :]' * vop * gc2[xi, yi, :]) * psik2_fft[xi, yi]
    end
    r_elem
    return abs(r_elem / Nsample)^2 * r1[2] * 2pi * 1.6 * 3.16 / 6.6 / 0.79^2 / pi / 0.02 * 1e2
end

function TMD_jx_6th(k; epsilonyy, sz, deltax=0.1)
    kx, ky = k
    h = deltax
    f(x) = TMD_hnn((x, ky); epsilonyy, sz)

    return -(f(kx - 3h) - 9f(kx - 2h) + 45f(kx - h) - 45f(kx + h) + 9f(kx + 2h) - f(kx + 3h)) / (60h)
end

function TMD_jy_6th(k; epsilonyy, sz, deltay=0.1)
    kx, ky = k
    h = deltay
    f(y) = TMD_hnn((kx, y); epsilonyy, sz)

    return -(f(ky - 3h) - 9f(ky - 2h) + 45f(ky - h) - 45f(ky + h) + 9f(ky + 2h) - f(ky + 3h)) / (60h)
end

function proj_elem(k, polarization; epsilonyy, sz)
    kx, ky = k
    px, py = polarization
    vop = TMD_jx_6th([kx, ky]; epsilonyy, sz) * px + TMD_jy_6th([kx, ky]; epsilonyy, sz) * py

    ham = TMD_hnn(k; epsilonyy, sz)
    eige, eigv = eigen(ham)
    vk, c2k, c1k = eachcol(eigv)
    Bloch_conduction1 = c1k
    Bloch_conduction2 = c2k
    Bloch_valence = vk
    Proj_c1 = c1k * c1k'
    Proj_c2 = c2k * c2k'
    Proj_v = vk * vk'
    return Proj_c2 * vop * Proj_v
end

function norm_elem(k, polarization; epsilonyy, sz)
    kx, ky = k
    px, py = polarization
    vop = TMD_jx_6th([kx, ky]; epsilonyy, sz) * px + TMD_jy_6th([kx, ky]; epsilonyy, sz) * py

    ham = TMD_hnn(k; epsilonyy, sz)
    eige, eigv = eigen(ham)
    vk, c2k, c1k = eachcol(eigv)
    Bloch_conduction1 = c1k
    Bloch_conduction2 = c2k
    Bloch_valence = vk
    Proj_c1 = c1k * c1k'
    Proj_c2 = c2k * c2k'
    Proj_v = vk * vk'
    return norm(c2k' * vop * vk)^2
end

function shiftvec(k, polarization; epsilonyy, sz)
    kx, ky = k

    deltax = 0.1
    h = deltax
    f(x) = proj_elem([x, ky], polarization; epsilonyy, sz)
    dx = -(f(kx - 3h) - 9f(kx - 2h) + 45f(kx - h) - 45f(kx + h) + 9f(kx + 2h) - f(kx + 3h)) / (60h)

    leftvec = proj_elem([kx, ky], polarization; epsilonyy, sz)
    shift_x = tr(leftvec' * dx)

    deltay = 0.1
    h = deltay
    g(y) = proj_elem([kx, y], polarization; epsilonyy, sz)
    dy = -(g(ky - 3h) - 9g(ky - 2h) + 45g(ky - h) - 45g(ky + h) + 9g(ky + 2h) - g(ky + 3h)) / (60h)

    leftvec = proj_elem([kx, ky], polarization; epsilonyy, sz)
    shift_y = tr(leftvec' * dy)

    return [shift_x, shift_y]
end



function proj_elem_higher(k, polarization; epsilonyy, sz)
    kx, ky = k
    px, py = polarization
    vop = TMD_jx_6th([kx, ky]; epsilonyy, sz) * px + TMD_jy_6th([kx, ky]; epsilonyy, sz) * py

    ham = TMD_hnn(k; epsilonyy, sz)
    eige, eigv = eigen(ham)
    vk, c2k, c1k = eachcol(eigv)
    Proj_c1 = c1k * c1k'
    Proj_c2 = c2k * c2k'
    Proj_v = vk * vk'
    return Proj_c1 * vop * Proj_v
end

function norm_elem_higher(k, polarization; epsilonyy, sz)
    kx, ky = k
    px, py = polarization
    vop = TMD_jx_6th([kx, ky]; epsilonyy, sz) * px + TMD_jy_6th([kx, ky]; epsilonyy, sz) * py

    ham = TMD_hnn(k; epsilonyy, sz)
    eige, eigv = eigen(ham)
    vk, c2k, c1k = eachcol(eigv)
    Proj_c1 = c1k * c1k'
    Proj_c2 = c2k * c2k'
    Proj_v = vk * vk'
    return norm(c1k' * vop * vk)^2
end

function shiftvec_higher(k, polarization; epsilonyy, sz)
    kx, ky = k

    deltax = 0.1
    h = deltax
    f(x) = proj_elem_higher([x, ky], polarization; epsilonyy, sz)
    dx = -(f(kx - 3h) - 9f(kx - 2h) + 45f(kx - h) - 45f(kx + h) + 9f(kx + 2h) - f(kx + 3h)) / (60h)

    leftvec = proj_elem_higher([kx, ky], polarization; epsilonyy, sz)
    shift_x = tr(leftvec' * dx)

    deltay = 0.1
    h = deltay
    g(y) = proj_elem_higher([kx, y], polarization; epsilonyy, sz)
    dy = -(g(ky - 3h) - 9g(ky - 2h) + 45g(ky - h) - 45g(ky + h) + 9g(ky + 2h) - g(ky + 3h)) / (60h)

    leftvec = proj_elem_higher([kx, ky], polarization; epsilonyy, sz)
    shift_y = tr(leftvec' * dy)

    return [shift_x, shift_y]
end

end