module exciton

using SpecialFunctions
using LinearAlgebra
using Plots, LaTeXStrings
using Struve
using KrylovKit
using SparseArrays
using ForwardDiff
using FFTW
using Measures
using TensorOperations

export GrapheneLattice, InteractionMatrix, Lattice2D, GrapheneBSE, add_bse_kernel, BlochStates, ham_index, add_wannier, envelope_real, Rshift_calculator, jxop, jyop,wrap_to_positive,wrap_to_pi

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

function wrap_to_positive(θ)
    return mod(θ, 2π)
end

function wrap_to_pi(θ)
    return mod(θ + π, 2π) - π
end

struct InteractionMatrix
	lattice::Lattice2D
	Nsample::Int
	VAA::Array{ComplexF64, 2}  # (Nsample*Nsample, Nsample*Nsample)
	VAB::Array{ComplexF64, 2}  # (Nsample*Nsample, Nsample*Nsample)
end

function InteractionMatrix(lattice::Lattice2D, Nsample::Int; lambda = 1, r0)
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

    screened_q_mat = zeros(ComplexF64, 2, Nsample, Nsample)
	labellist=[:AA, :AB]
	for (i, labeli) ∈ enumerate(labellist)
		screened_r_mat = zeros(ComplexF64, Nsample, Nsample)
		for x ∈ 1:Nsample
			for y ∈ 1:Nsample
				screened_r_mat[x, y]=screened_coulomb_R(x, y, labeli; cutoff, Lx = Nsample, Ly = Nsample, R1, R2, d1, r0)
			end
		end

		VR_roll = circshift(screened_r_mat, (1, 1))
		Vq_roll = fft(VR_roll)
        screened_q_mat[i,:,:]=circshift(Vq_roll, (-1, -1))
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

	if R > cutoff + 0.01
		return 0.0
	end

	return (struveh(0, R / r0) - bessely0(R / r0))
end

function HF_coulomb(kvec, kpvec; VAA, VAB, A = [0.0, 0.0], lattice, blochstates, Nsample)
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
	valence::Array{ComplexF64, 3}    # (Nsample, Nsample, Nvec)
	conduction::Array{ComplexF64, 3} # (Nsample, Nsample, Nvec)
	ecv::Array{Float64, 2}
end

struct WannierStates
	valence::Array{ComplexF64, 3}
	conduction::Array{ComplexF64, 3}
end

struct GrapheneBSE
	# Lattice info
	lattice::Lattice2D

	kappa::Vector{Float64}
	hopping::Vector{Float64}
	gap::ComplexF64
	Nsample::Int

	Bloch::BlochStates
	Wannier::Union{Nothing, WannierStates}
	BSEKernel::Union{Nothing, Array{ComplexF64, 2}}
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

	for i ∈ 1:Nsample
		for j ∈ 1:Nsample
			k = (i * b1 + j * b2) / Nsample
			eigensys, d = eigensystem(k; t1, t2, t3, delta = gap, A = kappa, R1, R2, lattice)
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

function compute_bse_kernel(sys::GrapheneBSE, VInt::InteractionMatrix, checkham = false)
	Nsample = sys.Nsample
	VAA = VInt.VAA
	VAB = VInt.VAB

	ham = zeros(ComplexF64, Nsample^2, Nsample^2)
	blochstates = sys.Bloch
	ecv = blochstates.ecv
	kappa = sys.kappa
	lattice = sys.lattice

	for kx ∈ 1:Nsample, ky ∈ 1:Nsample
		kidx = ham_index(kx, ky; xlength = Nsample, ylength = Nsample)
		ham[kidx, kidx] += ecv[kx, ky]
		for kxp ∈ 1:Nsample, kyp ∈ 1:Nsample
			kpidx = ham_index(kxp, kyp; xlength = Nsample, ylength = Nsample)
			ham[kidx, kpidx] += -HF_coulomb([kx, ky], [kxp, kyp]; VAA, VAB, A = kappa, lattice, blochstates, Nsample) / Nsample^2
		end
	end

	if checkham
		if !(isapprox(ham, ham'; rtol = 1e-6))
			println("warning!")
		end
	end

	return (ham + ham') / 2
end

function eigensystem(k::Vector; t1, t2, t3, delta, A = [0.0, 0.0], R1, R2, lattice)
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

function g_g(kx, ky; t1, t2, t3, delta, A = [0.0, 0.0], lattice)
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

function jxop(kx, ky; t1, t2, t3, delta, A = [0.0, 0.0], lattice)
	df = ForwardDiff.derivative(ax -> g_g(kx, ky; A = [ax, A[2]], t1, t2, t3, delta, lattice), A[1])
	return [0 conj(df); df 0]
end

function jyop(kx, ky; t1, t2, t3, delta, A = [0.0, 0.0], lattice)
	df = ForwardDiff.derivative(ay -> g_g(kx, ky; A = [A[1], ay], t1, t2, t3, delta, lattice), A[2])
	return [0 conj(df); df 0]
end

end
