using IJulia
# installkernel("Julia (3 threads)", env=Dict("JULIA_NUM_THREADS"=>"3"))
using Distributed
using Printf
using Statistics
using FFTW
using JLD2
using BenchmarkTools
using StaticArrays
using Profile
using LinearAlgebra
using DSP
using DelimitedFiles
using Dates
using PyPlot
using Fourier
using QuadGK
#using WindowFunctions
using LsqFit
using SparseArrays
using Random
using Statistics
#pygui(true)

function heaviside(vector)
    @inbounds for i = 1:length(vector)
        if vector[i] < 0
            vector[i] = 0.
        else
            vector[i] = 1.
        end
    end
    return vector
end
function door(n,i_deb,i_fin)
    dor = zeros(n)
    for i = i_deb:i_fin
        dor[i] = 1
    end
    return(dor)
end


function wind_hanning(vector, x_beg, x_end)
    n_beg = floor(Int32, x_beg/δx)
    n_end = floor(Int32, x_end/δx)

    fen = zeros(Complex{Float64}, length(vector))
    fen[n_beg:n_end] = vector[n_beg:n_end] .* DSP.Windows.hanning(n_end-n_beg+1)
    return(fen)
end

function energy(vector)
    I = 0
    for i = 1:n_x-1
        I += δk * vector[i]
    end
    return(I)
end

#Values of fondamental parameters of system
const ħ = 0.6582 #meV.ps
const ω_lp_0 = 1473.36/ħ
const γ_lp = 0.047/ħ
const g_lp = 0.0003/ħ
const m_lp = ħ^2/(2*1.29)

function save_vector(vector,file_name)
    open(folder*file_name, "w") do io
        @inbounds for i = 1:length(vector)
            writedlm(io, vector[i])
        end
    end
end

print("Go to parameters")