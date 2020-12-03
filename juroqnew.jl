"""
test points and known bases are all column based, which are contrary to the PyROQ version of row based.
"""

using Distributed
addprocs(nprocess)

using SharedArrays
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
    # u is a normalized vector
    dot(v, u) * u
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
#     while (mass_ns < 1.0 ||  mass_ns > 3.0 || mass_bh < 3.0)
    while (mass_ns < 1.0 ||  mass_ns > 3.0 || mass_bh < 1.0)
        # IMRPhenomNSBH waveform only works if mass_ns is in the 
        # range [1, 3]Msun and mass_bh > 1Msun.
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

function generate_a_random_waveform(deltaF, params_low, params_high, f_min, f_max, approximant)
    point = generate_a_param_point_NSBH(params_low, params_high)
    return generate_a_waveform_from_point(point, deltaF, f_min, f_max, approximant)
end

function generate_waveforms_from_points(points, deltaF, f_min, f_max, approximant)
    func(i) = generate_a_waveform_from_point(points[:, i], deltaF, f_min, f_max, approximant)
    waveforms = pmap(func, 1:size(points)[2])
    hcat(waveforms...)
end


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

function calculate_Bj(known_bases, emp_nodes)
    V = known_bases[emp_nodes, :]
    inverse_V = inv(V)
    Bj = known_bases * inverse_V
end
    
    
function empnodes(order, known_bases, known_points, Bj, emp_nodes, basis, point)
    
    if order=="quadratic"
        basis = abs.(basis) .^ 2
    end
    
    interpolantA = Bj * basis[emp_nodes]
        
    diff = interpolantA - basis
    r = abs.(diff)
    new_emp_node = argmax(r)
    
    if new_emp_node in emp_nodes
        println(current_time(), "--Adding no new basis because the new empirical node is duplicated.")
        return known_bases, known_points, Bj, emp_nodes
    end
    
    append!(emp_nodes, new_emp_node)
    sort!(emp_nodes)
    known_bases = hcat(known_bases, basis)
    known_points = hcat(known_points, point)
    
    Bj = calculate_Bj(known_bases, emp_nodes)
#     save("$(order)_cache.jld", "$(order)_bases", known_bases, "$(order)_points", known_points, "$(order)_Bj", Bj, "$(order)_emp_nodes", emp_nodes)   
    save("$(order)_cache.jld", "$(order)_points", known_points, "$(order)_emp_nodes", emp_nodes)   
    
    println(current_time(), "--Number of $(order) bases is $(size(known_bases, 2)).")
        
    return known_bases, known_points, Bj, emp_nodes
end

@everywhere function surroerror(order, Bj, emp_nodes, point, deltaF, f_min, f_max, approximant)
    
    hp_test = generate_a_waveform_from_point(point, deltaF, f_min, f_max, approximant)
    
    if order=="quadratic"
        hp_test = abs.(hp_test) .^ 2
    end
    
    
    interpolantA = Bj * hp_test[emp_nodes]
        
    surro = 2.0 * (1.0 - overlap_of_two_waveforms(hp_test, interpolantA))
end


# remove the test_points which have satisfied the tolerance requirement
function reduce_test_points(order, tolerance, Bj, emp_nodes, test_points, deltaF, f_min, f_max, approximant)
    if length(test_points)==0
        return test_points
    end
    
    println("--------------------------------------------------------------------------")
    println(current_time(), "--Start reducing $(length(test_points)) test points...")
    
    Bj0 = SharedArray(Bj)
    func(test_point) = surroerror(order, Bj0, emp_nodes, test_point, deltaF, f_min, f_max, approximant)
    each_time = @elapsed func(test_points[1])
    println(current_time(), "--Each point takes $(round(each_time, sigdigits=3)) seconds, it will take $(round(each_time * length(test_points)/3600, sigdigits=3)) hours if without parallelism.")
    println("--------------------------------------------------------------------------\n")
    flush(stdout)
    surro_errors = pmap(func, test_points)    
    Bj0 = nothing
    
    arg_newbasis = argmax(surro_errors) 
    new_point = test_points[arg_newbasis]
    
    masks = surro_errors .> tolerance  
#     masks[arg_newbasis] = false # rm new_point from the test_points to make sure no duplicate empiracal nodes issue
    new_test_points = test_points[masks]
    
    npzwrite("surro_errors_$(order).npy", surro_errors)
    
    failed_test_points_file = "failed_test_points_$(order).jld"
    save(failed_test_points_file, "failed_test_points", new_test_points) 
    
    @everywhere gc()
    println("--------------------------------------------------------------------------")
    println(current_time(), "--\033[91m Number of test points has been reduced from $(length(test_points)) to $(length(new_test_points)).\033[0m")
    println("-------------------------------------------------------------------------- \n")
    return new_test_points, new_point
end

function run_roq(order, tolerance, npts, params_low, params_high, deltaF, f_min, f_max, approximant; resume=true, ntests)
    println("--------------------------------------------------------------------------")    
    println(current_time(), "--\033[91m The $(order) tolerance is set to be $(tolerance). \033[0m")
    println("--------------------------------------------------------------------------\n")
    flush(stdout)
        
    freq = f_min:deltaF:(f_max-deltaF) 
    
    test_points_file = "test_points_$(order).jld"
    if isfile(test_points_file)
        caches = load(test_points_file)
        test_points = caches["test_points"]
        println(current_time(), "\033[91m Finish loading test points. \033[0m")
    else
        println(current_time(), "--\033[91m Start generating $(ntests) random test points... \033[0m")
        test_points = generate_params_points(ntests, params_low, params_high)   
        save(test_points_file, "test_points", test_points) 
        println(current_time(), "--\033[91m Finish generating test points and start building $(order) bases. \033[0m") 
    end     
    
    failed_test_points_file = "failed_test_points_$(order).jld"
    if isfile(failed_test_points_file)
        caches = load(failed_test_points_file)
        failed_test_points = caches["failed_test_points"]
    else
        failed_test_points = []
    end
    
    if resume && isfile("$(order)_cache.jld")
        caches = load("$(order)_cache.jld")
        known_points = caches["$(order)_points"]
        emp_nodes = caches["$(order)_emp_nodes"]  
        known_bases = generate_waveforms_from_points(known_points, deltaF, f_min, f_max, approximant)
#         known_bases = caches["$(order)_bases"]
        Bj = calculate_Bj(known_bases, emp_nodes)        
        println(current_time(), "--Resume from $(size(known_bases, 2)) known $(order) bases...")
    else
        known_points_start = test_points[1]
        known_points = hcat([known_points_start]...)
        
        hp1 = generate_a_waveform_from_point(known_points_start, deltaF, f_min, f_max, approximant)
        if order == "quadratic"
            hp1 = abs.(hp1) .^ 2
        end
        
        known_bases_start = normalize(hp1)
        known_bases = hcat([known_bases_start]...)
        emp_nodes = [argmax(abs.(known_bases_start))]
        Bj_start = known_bases_start/known_bases_start[emp_nodes[1]]
        Bj = hcat([Bj_start]...)
    end
    
    println(current_time(), "--Number of $(order) bases is $(size(known_bases, 2)).")    
    
    for new_test_points in [failed_test_points, test_points[1:3*10^3], test_points[1:10^4], test_points[1:10^5], test_points[10^5:3*10^5], test_points[3*10^5:end]]        
        while length(new_test_points) > 0
            new_test_points, new_point = reduce_test_points(order, tolerance, Bj, emp_nodes, new_test_points, deltaF, f_min, f_max, approximant)
            if length(new_test_points)>0
                new_basis = generate_a_waveform_from_point(new_point, deltaF, f_min, f_max, approximant)
                known_bases, known_points, Bj, emp_nodes = empnodes(order, known_bases, known_points, Bj, emp_nodes, new_basis, new_point)
            end
        end
    end
                
    for i in 1:10
        new_test_points, new_point = reduce_test_points(order, tolerance, Bj, emp_nodes, test_points, deltaF, f_min, f_max, approximant)
        if length(new_test_points)==0
            break
        else
            new_basis = generate_a_waveform_from_point(new_point, deltaF, f_min, f_max, approximant)
            known_bases, known_points, Bj, emp_nodes = empnodes(order, known_bases, known_points, Bj, emp_nodes, new_basis, new_point)
            while length(new_test_points) > 0
                new_test_points, new_point = reduce_test_points(order, tolerance, Bj, emp_nodes, new_test_points, deltaF, f_min, f_max, approximant)
                if length(new_test_points) > 0
                    new_basis = generate_a_waveform_from_point(new_point, deltaF, f_min, f_max, approximant)
                    known_bases, known_points, Bj, emp_nodes = empnodes(order, known_bases, known_points, Bj, emp_nodes, new_basis, new_point) 
                end
            end
        end
    end
    
           
    b_matrix = transpose(Bj)
    fnodes = freq[emp_nodes]
    
    npzwrite("B_$(order).npy", b_matrix)
    npzwrite("fnodes_$(order).npy", fnodes)
    
    println(current_time(), "--Finish building $(order) basis.")
    println("--------------------------------------------------------------------------")
    println(current_time(), "--\033[91m Number of $(order) basis elements is $(size(known_bases, 2)) and the $(order) ROQ data are saved. \033[0m")
    println("--------------------------------------------------------------------------\n")
    flush(stdout)
end