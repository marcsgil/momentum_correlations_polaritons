using HDF5

function write_parameters!(parent, param)
    for (key, value) in zip(keys(param), values(param))
        attrs(parent)[string(key)] = value
    end
end

function read_parameters(parent)
    ks = Symbol.(keys(attrs(parent)))
    vals = values(attrs(parent))
    (; (ks .=> vals)...)
end

function reset_simulation!(saving_path, group_name)
    h5open(saving_path, "cw") do file
        group = file[group_name]
        group["one_point_r"][:, :, :, :] = 0
        group["two_point_r"][:, :] = 0
        group["one_point_k"][:, :, :, :] = 0
        group["two_point_k"][:, :] = 0
        group["n_ave"][:] = 0
    end
    nothing
end