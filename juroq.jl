using Distributed
addprocs(nprocess)

@everywhere using SharedArrays
using BenchmarkTools
@everywhere using LinearAlgebra
@everywhere using Distributions
@everywhere using Dates
using JLD
using NPZ
@everywhere using PyCall
@everywhere ls = pyimport("lalsimulation")
@everywhere lal = pyimport("lal")

@everywhere gc() = Base.GC.gc()
@everywhere current_time() = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

@everywhere norm(v) = sqrt(dot(v, v))

@everywhere LinearAlgebra.normalize(v::Array{Complex{Float64},1}) = v/norm(v)

# Calculating the projection of complex vector v on complex vector u
@everywhere function proj(u, v)
    # notice: this algrithm assume denominator isn't zero
    temp = dot(v, u) / dot(u, u) 
    temp * u
end

# Calculating the normalized residual (= a new basis) of a vector vec from known bases
@everywhere function gram_schmidt(bases, vec)
    for i in 1:size(bases, 2)
        vec -= proj(bases[:, i], vec)
    end
    return normalize(vec) # normalized new basis
end

# Calculating overlap of two waveforms
@everywhere function overlap_of_two_waveforms(wf1, wf2)
    wf1norm = normalize(wf1) # normalize the first waveform
    wf2norm = normalize(wf2) # normalize the second waveform
#     diff = wf1norm - wf2norm
    #overlap = 1 - 0.5*(np.vdot(diff,diff))
    overlap = real(dot(wf1norm, wf2norm))
end

@everywhere function spherical_to_cartesian(sph)
    #x = sph[0]*np.sin(sph[1])*np.cos(sph[2])
    #y = sph[0]*np.sin(sph[1])*np.sin(sph[2])
    #z = sph[0]*np.cos(sph[1])
    #car = [x,y,z]
    return sph #car
end

@everywhere function massrange(mc_low, mc_high, q_low, q_high)
    mmin = get_m1m2_from_mcq(mc_low,q_high)[2]
    mmax = get_m1m2_from_mcq(mc_high,q_high)[1]
    return [mmin, mmax]
end

@everywhere function get_m1m2_from_mcq(mc, q)
    m2 = mc * q^(-0.6) * (1+q)^0.2
    m1 = m2 * q
    return [m1, m2]
end


@everywhere generate_a_rand_number(a) = a[1]==a[2] ? a[1] : rand(Uniform(a[1], a[2]))
@everywhere generate_a_rand_point(params_low, params_high) = [generate_a_rand_number(a) for a in zip(params_low, params_high)]

@everywhere function generate_a_param_point_NSBH(params_low, params_high)
    
    mass_ns = 0.0 # initial mass of neutron star
    mass_bh = 0.0 # initial mass of black hole
    point = zeros(length(params_low))
    while (mass_ns < 1.0 ||  mass_ns > 3.0 || mass_bh < 3.0)
        # IMRPhenomNSBH waveform only works if mass_ns is in the 
        # range [1, 3]Msun and mass_bh > 3Msun.
        point = generate_a_rand_point(params_low, params_high)
        mc = point[1]
        q = point[2]
        mass_bh, mass_ns = get_m1m2_from_mcq(mc, q) 
    end
    return point
end

@everywhere generate_params_points(npts, params_low, params_high) = [generate_a_param_point_NSBH(params_low, params_high) for i in 1:npts]

@everywhere function generate_a_waveform_from_mcq(mc, q, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef, deltaF, f_min, f_max, approximant)
    waveFlags = lal.CreateDict()
    f_ref = 20
    
    m1, m2 = get_m1m2_from_mcq(mc,q)
    mass1 = m1 * lal.lal.MSUN_SI
    mass2 = m2 * lal.lal.MSUN_SI
    ls.SimInspiralWaveformParamsInsertTidalLambda1(waveFlags, lambda1)
    ls.SimInspiralWaveformParamsInsertTidalLambda2(waveFlags, lambda2) 
    
    distance = 10 * lal.lal.PC_SI * 1.0e6  # 10 Mpc is default 
    
    hp, hc = ls.SimInspiralChooseFDWaveform(mass1, mass2, spin1..., spin2..., 
        distance, iota, phiRef, 0, ecc, 0, deltaF, f_min, f_max, f_ref, 
        waveFlags, approximant)
    return hp.data.data[Int(f_min/deltaF)+1:Int(f_max/deltaF)]
end

@everywhere function generate_a_waveform_from_point(point, deltaF, f_min, f_max, approximant)
    if approximant==ls.IMRPhenomNSBH
        mc =  point[1]
        q = point[2]
        s1 = spherical_to_cartesian(point[3:5])
        s2 = spherical_to_cartesian(point[6:8])   
        iota = point[9]
        phiref = point[10]
        lambda1 = point[11]
        lambda2 = point[12]
        ecc = 0
    end
    
    hp = generate_a_waveform_from_mcq(
        mc, q, s1, s2, ecc, lambda1, lambda2, iota, phiref, 
        deltaF, f_min, f_max, approximant)
end

@everywhere function generate_a_random_waveform(deltaF, params_low, params_high, f_min, f_max, approximant)
    point = generate_a_param_point_NSBH(params_low, params_high)
    return generate_a_waveform_from_point(point, deltaF, f_min, f_max, approximant)
end

@everywhere function generate_waveforms_from_points(points, deltaF, f_min, f_max, approximant)
    waveforms = [generate_a_waveform_from_point(point, deltaF, f_min, f_max, approximant) for point in points]
    hcat(waveforms...)
end

# @everywhere function parameter_number(approximant)
#     if approximant in [ls.IMRPhenomPv2, ls.IMRPhenomPv3, ls.IMRPhenomPv3HM, ls.IMRPhenomXHM]
#         nparams = 10
#     elseif approximant in [ls.TaylorF2Ecc]
#         nparams = 11
#     elseif approximant in [ls.IMRPhenomPv2_NRTidal, ls.IMRPhenomNSBH]
#         nparams = 12
#     else
#         println("This waveform hasn't been implemented.")
#         return
#     end
# end

@everywhere function param_range(mc_low, mc_high, q_low, q_high, s1sphere_low, s1sphere_high, s2sphere_low, s2sphere_high, ecc_low, ecc_high, lambda1_low, lambda1_high, lambda2_low, lambda2_high, iota_low, iota_high, phiref_low, phiref_high, approximant)
    if approximant in [ls.IMRPhenomPv2, ls.IMRPhenomPv3, ls.IMRPhenomPv3HM, ls.IMRPhenomXHM]
        params_low = [mc_low, q_low, s1sphere_low..., s2sphere_low..., iota_low, phiref_low] 
        params_high = [mc_high, q_high, s1sphere_high..., s2sphere_high..., iota_high, phiref_high]
    
    elseif approximant in [ls.TaylorF2Ecc]
        params_low = [mc_low, q_low, s1sphere_low..., s2sphere_low..., iota_low, phiref_low, ecc_low] 
        params_high = [mc_high, q_high, s1sphere_high..., s2sphere_high..., iota_high, phiref_high, ecc_high]
        
    elseif approximant in [ls.IMRPhenomPv2_NRTidal, ls.IMRPhenomNSBH]
        params_low = [mc_low, q_low, s1sphere_low..., s2sphere_low..., iota_low, phiref_low, lambda1_low, lambda2_low]
        params_high = [mc_high, q_high, s1sphere_high..., s2sphere_high..., iota_high, phiref_high, lambda1_high, lambda2_high]
    else
        println("This waveform hasn't been implemented.")
        return
    end
    
    return params_low, params_high
end

@everywhere function empnodes(order, known_bases, known_points, inverse_V, emp_nodes, basis, point)
    
    ndim = size(known_bases, 2)
    Ci = inverse_V * basis[emp_nodes]
    
    interpolantA = zeros(Complex, length(basis))
    for j in 1:ndim
        interpolantA += Ci[j] * known_bases[:, j]
    end  
        
    diff = interpolantA - basis
    r = abs.(diff)
    new_emp_node = argmax(r)
    
    if new_emp_node in emp_nodes
        println(current_time(), "--Adding no new basis because the new empirical node is duplicated.")
        return known_bases, known_points, inverse_V, emp_nodes
    end
    
    append!(emp_nodes, new_emp_node)
    sort!(emp_nodes)
    known_bases = hcat(known_bases, basis)
    known_points = hcat(known_points, point)
        
    V = known_bases[emp_nodes, :]
    inverse_V = inv(V)
#     inverse_V = np.linalg.pinv(V)
    save("$(order)_cache.jld", "$(order)_bases", known_bases, "$(order)_points", known_points, "$(order)_inverse_V", inverse_V, "$(order)_emp_nodes", emp_nodes)    
    
    println(current_time(), "--The number of $(order) bases is $(size(known_bases, 2)).")
        
    return known_bases, known_points, inverse_V, emp_nodes
end

@everywhere function surroerror(order, known_bases, inverse_V, emp_nodes, point, deltaF, f_min, f_max, approximant)
    
    hp_test = generate_a_waveform_from_point(point, deltaF, f_min, f_max, approximant)
    
    if order=="quadratic"
        hp_test = abs.(hp_test) .^ 2
    end
    
    ndim = size(known_bases, 2)
    Ci = inverse_V * hp_test[emp_nodes]
    interpolantA = zeros(Complex, length(hp_test))
    for j in 1:ndim
        interpolantA += Ci[j] * known_bases[:, j]
    end 
        
    surro = (1 - overlap_of_two_waveforms(hp_test, interpolantA))*deltaF
end


@everywhere function compute_modulus(order, point, deltaF, f_min, f_max, approximant, known_bases)
    hp_tmp = generate_a_waveform_from_point(point, deltaF, f_min, f_max, approximant) # data_tmp is hplus and is a complex vector 
    if order=="quadratic"
        hp_tmp = abs.(hp_tmp) .^ 2
    end
    
    residual = hp_tmp
    for i in 1:size(known_bases, 2)
        residual -= proj(known_bases[:, i], hp_tmp)
    end
    real(norm(residual))
end 

function search_a_new_basis(order, known_bases, npts, params_low, params_high, deltaF, f_min, f_max, approximant)
    
    known_bases0 = SharedArray(known_bases)
    
    points = generate_params_points(npts, params_low, params_high)
    
    func(point) = compute_modulus(order, point, deltaF, f_min, f_max, approximant, known_bases0)
    modula = pmap(func, points)
    
    arg_newbasis = argmax(modula)
    new_point = points[arg_newbasis]
    hp_new = generate_a_waveform_from_point(new_point, deltaF, f_min, f_max, approximant)
    
    if order == "quadratic"
        hp_new = abs.(hp_new) .^ 2  
    end
    new_basis = gram_schmidt(known_bases0, hp_new)    
    flush(stdout)
    
    known_bases0 = nothing
    @everywhere gc()
    return new_point, new_basis # elements, masses&spins, residual mod  
end

# add new bases to the known_bases such that points satisfy the tolerance requirement
function add_bases(order, tolerance, known_bases, known_points, inverse_V, emp_nodes, npts, params_low, params_high, deltaF, f_min, f_max, approximant, points)
    for i in 1:length(points)
        test_point = points[i]        
        surro = surroerror(order, known_bases, inverse_V, emp_nodes, test_point, deltaF, f_min, f_max, approximant)
        while surro > tolerance
            println(current_time(), "--The surrogate error is $(round(surro, sigdigits=3)) for the $(i)th test waveform, rebuilding...\n")
            new_point, new_basis = search_a_new_basis(order, known_bases, npts, params_low, params_high, deltaF, f_min, f_max, approximant)
            known_bases, known_points, inverse_V, emp_nodes = empnodes(order, known_bases, known_points, inverse_V, emp_nodes, new_basis, new_point)
            surro = surroerror(order, known_bases, inverse_V, emp_nodes, test_point, deltaF, f_min, f_max, approximant)
            @everywhere gc()
        end
    end
     
    println("--------------------------------------------------------------------------")
    println(current_time(), "--\033[91m Finish adding new $(order) bases to pass $(length(points)) test waveforms.\033[0m")
    println("--------------------------------------------------------------------------\n")
    flush(stdout)
    return known_bases, known_points, inverse_V, emp_nodes
end

# remove the test_points which have satisfied the tolerance requirement
function reduce_test_points(order, tolerance, known_bases, inverse_V, emp_nodes, test_points, deltaF, f_min, f_max, approximant)
    if length(test_points)==0
        return test_points
    end
    
    println("--------------------------------------------------------------------------")
    println(current_time(), "--Start reducing $(length(test_points)) test points...")
    
    known_bases0 = SharedArray(known_bases)
    inverse_V0 = SharedArray(inverse_V)
    func(test_point) = surroerror(order, known_bases0, inverse_V0, emp_nodes, test_point, deltaF, f_min, f_max, approximant)
    func(test_points[1]); each_time = @elapsed func(test_points[1])
    println(current_time(), "--Each point takes $(round(each_time, sigdigits=3)) seconds, it will take $(round(each_time * length(test_points)/3600, sigdigits=3)) hours if without parallelism.")
    println("--------------------------------------------------------------------------\n")
    flush(stdout)
    surro_errors = pmap(func, test_points)    
    known_bases0 = nothing
    inverse_V0 = nothing
    
    masks = surro_errors .> tolerance
    
    new_test_points = test_points[masks]
    @everywhere gc()
    println("--------------------------------------------------------------------------")
    println(current_time(), "--\033[91m Number of test_points has been reduced from $(length(test_points)) to $(length(new_test_points)).\033[0m")
    println("-------------------------------------------------------------------------- \n")
    return new_test_points  
end

function run_roq(order, tolerance, npts, params_low, params_high, deltaF, f_min, f_max, approximant; resume=true, ntests)
    println("--------------------------------------------------------------------------")
    println(current_time(), "--Start generating $(ntests) random test points...")
    test_points = generate_params_points(ntests, params_low, params_high)    
    println(current_time(), "--Finish generating test points and start building $(order) bases.") 
    println("--------------------------------------------------------------------------\n")
    flush(stdout)
    
    freq = f_min:deltaF:(f_max-deltaF) # need to be checked
       
    if resume && isfile("$(order)_cache.jld")
        caches = load("$(order)_cache.jld")
        known_bases = caches["$(order)_bases"]
        known_points = caches["$(order)_points"]
        inverse_V = caches["$(order)_inverse_V"]
        emp_nodes = caches["$(order)_emp_nodes"]       
        println(current_time(), "--Resume from $(size(known_bases, 2)) known $(order) bases...")
    else
        known_points_start = generate_a_param_point_NSBH(params_low, params_high)
        known_points = hcat([known_points_start]...)
        
        hp1 = generate_a_waveform_from_point(known_points_start, deltaF, f_min, f_max, approximant)
        if order == "quadratic"
            hp1 = abs.(hp1) .^ 2
        end
        
        known_bases_start = normalize(hp1)
        known_bases = hcat([known_bases_start]...)
        emp_nodes = [argmax(abs.(known_bases_start))]
        inverse_V = Array{Complex{Float64}, 2}(undef, 1, 1)
        inverse_V[1, 1] = 1.0/known_bases_start[emp_nodes[1]]
    end
    
    println(current_time(), "--The number of $(order) bases is $(size(known_bases, 2)).")
    
    n_test_points = length(test_points)
    while n_test_points >= 100
        points = test_points[1:100]
        test_points = test_points[101:end]
        
        known_bases, known_points, inverse_V, emp_nodes = add_bases(order, tolerance, known_bases, known_points, inverse_V, emp_nodes, npts, params_low, params_high, deltaF, f_min, f_max, approximant, points)                
        
        test_points = reduce_test_points(order, tolerance, known_bases, inverse_V, emp_nodes, test_points, deltaF, f_min, f_max, approximant)
        n_test_points = length(test_points)
    end
                
    if n_test_points > 0
        known_bases, known_points, inverse_V, emp_nodes = add_bases(order, tolerance, known_bases, known_points, inverse_V, emp_nodes, npts, params_low, params_high, deltaF, f_min, f_max, approximant, test_points)  
    end
    
           
    b_matrix = known_bases * inverse_V
    fnodes = freq[emp_nodes]
    
    npzwrite("B_$(order).npy", transpose(b_matrix))
    npzwrite("fnodes_$(order).npy", fnodes)
    println(current_time(), "--Finish building $(order) basis.")
    println("--------------------------------------------------------------------------")
    println(current_time(), "--\033[91m Number of $(order) basis elements is $(size(known_bases, 2)) and the $(order) ROQ data are saved. \033[0m")
    println("--------------------------------------------------------------------------\n")
    flush(stdout)
end
            
