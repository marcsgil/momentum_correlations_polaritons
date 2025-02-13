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