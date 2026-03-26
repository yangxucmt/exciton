module exciton

using SpecialFunctions
using LinearAlgebra, Base.Threads
using Plots, LaTeXStrings
using Struve
using KrylovKit
using SparseArrays
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

end