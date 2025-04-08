using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--max_datetime"
        help = "an option with an argument"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg, val) in parsed_args
        println(arg)
        println("  $arg  =>  $val")
    end

    t = Tuple(parse.(Int, split(parsed_args["max_datetime"], ',')))
    @show t
end

main()