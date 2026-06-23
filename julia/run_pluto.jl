#!/usr/bin/env julia

# Launch Pluto on the Numeria notebook.
#
#   julia julia/run_pluto.jl              # from a dev shell, in the repo root
#   nix run .#pluto                       # via the flake app
#
# Pluto and the notebook's own package dependencies are installed on first run;
# the notebook (Meetup_Numeria.jl) embeds its own Project/Manifest.

import Pkg

# Make sure Pluto is available in the launching environment, installing it once
# into the default environment if it is missing.
try
    @eval import Pluto
catch
    Pkg.add("Pluto")
    @eval import Pluto
end

# Resolve the notebook to open: an explicit CLI argument wins, otherwise look
# next to this script (dev-shell use) and then under the current directory (so
# `nix run`, where this script lives in the store, opens the working copy).
function find_notebook()
    isempty(ARGS) || return abspath(ARGS[1])
    candidates = [
        joinpath(@__DIR__, "Meetup_Numeria.jl"),
        joinpath(pwd(), "julia", "Meetup_Numeria.jl"),
        joinpath(pwd(), "Meetup_Numeria.jl"),
    ]
    for candidate in candidates
        isfile(candidate) && return candidate
    end
    return nothing
end

notebook = find_notebook()
if notebook === nothing
    @warn "Meetup_Numeria.jl not found near the script or in the current directory; opening Pluto without a notebook."
    Pluto.run()
else
    Pluto.run(; notebook)
end
