### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b0000000-0000-0000-0000-000000000003
begin
    using Plots
    using DataFrames
    using PlutoUI
    using Random
    using Statistics
    using Combinatorics
    using StaticArrays
end

# ╔═╡ b0000000-0000-0000-0000-000000000001
md"""
# Outbreak in Numeria: The Null Fever Simulation

The city of **Numeria** is isolated for the next **180 days**. The mountain passes are closed, and the council must choose policies before outside medical caravans can arrive. A new disease, **Null Fever**, is spreading through the city.

Run the **setup cells** first. After that, each mission can be run separately.

> This is a [Pluto.jl](https://plutojl.org) port of the original Jupyter notebook. Pluto is *reactive*: when you change a cell or drag a slider, every dependent cell re-runs automatically, and cell order does not matter.
"""

# ╔═╡ b0000000-0000-0000-0000-000000000002
md"## Setup 1 — Libraries"

# ╔═╡ b0000000-0000-0000-0000-000000000004
md"""
## Setup 2 — Model, numerical methods, and helper functions

The epidemic state is

```math
Y(t)=(S(t),V(t),E(t),I(t),Q(t),R(t),C(t)).
```

The variable $C(t)$ is cumulative infections.
"""

# ╔═╡ b0000000-0000-0000-0000-000000000005
begin
    const COMPARTMENTS = (:S, :V, :E, :I, :Q, :R, :C)
    const IDX = NamedTuple{COMPARTMENTS}(ntuple(i -> i, length(COMPARTMENTS)))
    const POPULATION_IDXS = [IDX.S, IDX.V, IDX.E, IDX.I, IDX.Q, IDX.R]
end

# ╔═╡ b0000000-0000-0000-0000-000000000006
begin
    @kwdef struct Params
        N::Float64 = 100_000.0
        beta0::Float64 = 0.75
        a::Float64 = 0.10
        kappa::Float64 = 25.0
        rho::Float64 = 0.05
        eta::Float64 = 0.80
        nu::Float64 = 0.0005
        omega::Float64 = 1.0 / 240.0
        sigma::Float64 = 1.0 / 4.0
        gamma::Float64 = 1.0 / 8.0
        delta::Float64 = 0.08
        gamma_q::Float64 = 1.0 / 10.0
    end

    "Return a copy of `p` with the given fields replaced (the analogue of Python's `replace`)."
    function reparam(p::Params; changes...)
        fields = Dict(f => getfield(p, f) for f in fieldnames(Params))
        Params(; merge(fields, changes)...)
    end

    initial_state() = SVector(99_940.0, 0.0, 40.0, 20.0, 0.0, 0.0, 60.0)
end

# ╔═╡ b0000000-0000-0000-0000-000000000007
begin
    function beta_eff(t, y, p::Params)
        infectious = y[IDX.I]
        weekly_factor = 1.0 + p.a * sin(2.0 * pi * t / 7.0)
        behavior_factor = 1.0 + p.kappa * infectious / p.N
        return p.beta0 * weekly_factor / behavior_factor
    end

    function force_of_infection(t, y, p::Params)
        infectious = y[IDX.I]
        isolated = y[IDX.Q]
        return beta_eff(t, y, p) * (infectious + p.rho * isolated) / p.N
    end

    function rhs(t, y, p::Params)
        S, V, E, I, Q, R, C = y
        lam = force_of_infection(t, y, p)

        infections_from_S = lam * S
        infections_from_V = (1.0 - p.eta) * lam * V
        incidence = infections_from_S + infections_from_V

        dS = -infections_from_S - p.nu * S + p.omega * R
        dV = p.nu * S - infections_from_V
        dE = incidence - p.sigma * E
        dI = p.sigma * E - p.gamma * I - p.delta * I
        dQ = p.delta * I - p.gamma_q * Q
        dR = p.gamma * I + p.gamma_q * Q - p.omega * R
        dC = incidence

        return SVector(dS, dV, dE, dI, dQ, dR, dC)
    end

    conservation_defect(t, y, p::Params) = sum(rhs(t, y, p)[POPULATION_IDXS])
end

# ╔═╡ b0000000-0000-0000-0000-000000000008
begin
    # A numerical method is just a function that maps the current state Y_n to Y_{n+1}.
    # We pass that function to `simulate` as a value, so adding a method = adding a function.

    "Explicit Euler step."
    euler_step(t, y, dt, p::Params) = y .+ dt .* rhs(t, y, p)

    "Heun's method (second-order predictor-corrector) step."
    function heun_step(t, y, dt, p::Params)
        k1 = rhs(t, y, p)
        k2 = rhs(t + dt, y .+ dt .* k1, p)
        return y .+ 0.5 .* dt .* (k1 .+ k2)
    end

    "Classic fourth-order Runge-Kutta step."
    function rk4_step(t, y, dt, p::Params)
        k1 = rhs(t, y, p)
        k2 = rhs(t + 0.5 * dt, y .+ 0.5 .* dt .* k1, p)
        k3 = rhs(t + 0.5 * dt, y .+ 0.5 .* dt .* k2, p)
        k4 = rhs(t + dt, y .+ dt .* k3, p)
        return y .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
    end

    # Display names for the stepper functions, used in plot titles and table columns.
    const METHOD_LABELS = Dict(euler_step => "Euler", heun_step => "Heun", rk4_step => "RK4")
    method_label(step) = METHOD_LABELS[step]

    "Simulate from t=0 to T, advancing with the `method` stepper function."
    function simulate(
        p::Params = Params();
        method = rk4_step,
        dt::Real = 0.25,
        T::Real = 180.0,
        y0 = nothing,
    )
        y_init =
            y0 === nothing ? initial_state() :
            SVector{length(COMPARTMENTS),Float64}(y0)

        nsteps = ceil(Int, T / dt)
        t = Vector{Float64}(undef, nsteps + 1)
        Y = Matrix{Float64}(undef, nsteps + 1, length(COMPARTMENTS))
        t[1] = 0.0
        Y[1, :] = y_init

        current_t = 0.0
        current_y = y_init
        for n = 1:nsteps
            h = min(float(dt), T - current_t)
            current_y = method(current_t, current_y, h, p)
            current_t += h
            t[n+1] = current_t
            Y[n+1, :] = current_y
        end
        return t, Y
    end
end

# ╔═╡ b0000000-0000-0000-0000-000000000009
begin
    hospital_load(Y) = 0.04 .* Y[:, IDX.I] .+ 0.10 .* Y[:, IDX.Q]

    population(Y) = vec(sum(Y[:, POPULATION_IDXS]; dims = 2))

    function summarize(t, Y, p::Params = Params(); hospital_capacity = 600.0)
        infectious = Y[:, IDX.I]
        H = hospital_load(Y)
        pop_error = maximum(abs.(population(Y) .- p.N))
        overload_days =
            length(t) > 1 ? sum((H[1:(end-1)] .> hospital_capacity) .* diff(t)) : 0.0
        return (
            peak_I = maximum(infectious),
            day_peak_I = t[argmax(infectious)],
            peak_hospital_load = maximum(H),
            day_peak_hospital_load = t[argmax(H)],
            overload_days = overload_days,
            final_cumulative_cases = Y[end, IDX.C],
            minimum_compartment = minimum(Y[:, POPULATION_IDXS]),
            max_population_error = pop_error,
        )
    end
end

# ╔═╡ b0000000-0000-0000-0000-00000000000a
begin
    "Piecewise-linear interpolation, the analogue of numpy's `interp` (x must be ascending)."
    function interp1(xq, x, y)
        n = length(x)
        map(xq) do q
            if q <= x[1]
                y[1]
            elseif q >= x[n]
                y[n]
            else
                k = searchsortedlast(x, q)
                frac = (q - x[k]) / (x[k+1] - x[k])
                y[k] + frac * (y[k+1] - y[k])
            end
        end
    end

    interp_component(t_source, Y_source, t_target, name) =
        interp1(t_target, t_source, Y_source[:, IDX[name]])

    function error_against_reference(t, Y, t_ref, Y_ref, name = :I)
        ref = interp_component(t_ref, Y_ref, t, name)
        return maximum(abs.(Y[:, IDX[name]] .- ref))
    end
end

# ╔═╡ b0000000-0000-0000-0000-00000000000b
begin
    function calibration_loss(beta0, kappa, weeks, observed_I, base::Params)
        q = reparam(base; beta0 = Float64(beta0), kappa = Float64(kappa))
        t, Y = simulate(q; method = rk4_step, dt = 0.5, T = 180.0)
        pred_I = interp_component(t, Y, weeks, :I)
        return mean((pred_I .- observed_I) .^ 2)
    end

    function make_weekly_observations(base::Params; seed = 7)
        rng = MersenneTwister(seed)
        true_params = reparam(base; beta0 = 0.78, kappa = 18.0)
        t_true, Y_true = simulate(true_params; method = rk4_step, dt = 0.1, T = 180.0)
        weeks = collect(7.0:7.0:180.0)
        true_I = interp_component(t_true, Y_true, weeks, :I)
        noise = 250.0 .* randn(rng, length(weeks))
        observed_I = max.(true_I .+ noise, 0.0)
        return weeks, observed_I, true_params
    end
end

# ╔═╡ b0000000-0000-0000-0000-00000000000c
begin
    policy_cards() = [
        (name = "vaccine sprint", cost = 2, apply = q -> reparam(q; nu = q.nu + 0.0007)),
        (
            name = "booster campaign",
            cost = 2,
            apply = q -> reparam(q; eta = min(0.95, q.eta + 0.08)),
        ),
        (
            name = "rapid isolation",
            cost = 2,
            apply = q -> reparam(q; delta = q.delta + 0.05),
        ),
        (name = "public alerts", cost = 1, apply = q -> reparam(q; kappa = q.kappa + 15.0)),
        (
            name = "mask distribution",
            cost = 1,
            apply = q -> reparam(q; beta0 = 0.95 * q.beta0),
        ),
        (name = "isolation support", cost = 1, apply = q -> reparam(q; rho = 0.5 * q.rho)),
    ]

    function apply_policy_set(base, chosen)
        q = base
        names = String[]
        cost = 0
        for card in chosen
            q = card.apply(q)
            push!(names, card.name)
            cost += card.cost
        end
        return q, names, cost
    end
end

# ╔═╡ b0000000-0000-0000-0000-00000000000d
begin
    function plot_compartments(t, Y; title = "Compartment evolution")
        plt =
            plot(; title, xlabel = "Time in days", ylabel = "People", legend = :outerright)
        for name in (:S, :V, :E, :I, :Q, :R)
            plot!(plt, t, Y[:, IDX[name]]; label = string(name))
        end
        return plt
    end

    function plot_dashboard(t, Y; title = "City dashboard", hospital_capacity = 600.0)
        plt = plot(
            t,
            Y[:, IDX.I];
            label = "Infectious I(t)",
            title,
            xlabel = "Time in days",
            ylabel = "People / beds",
            legend = :topright,
        )
        plot!(plt, t, Y[:, IDX.Q]; label = "Isolated Q(t)")
        plot!(plt, t, hospital_load(Y); label = "Hospital load H(t)")
        hline!(plt, [hospital_capacity]; label = "Hospital capacity", ls = :dash)
        return plt
    end
end

# ╔═╡ b0000000-0000-0000-0000-00000000000e
base_params = Params()

# ╔═╡ b0000000-0000-0000-0000-00000000000f
md"""
### 🎛️ Interactive explorer

Drag the controls — every dependent cell below updates automatically. This is what Pluto adds over a classic notebook.

| control | value |
|---|---|
| Method | $(@bind ex_method Select([rk4_step => "RK4", heun_step => "Heun", euler_step => "Euler"], default = rk4_step)) |
| Δt (days) | $(@bind ex_dt Slider([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], default = 0.25, show_value = true)) |
| β₀ | $(@bind ex_beta0 Slider(0.4:0.05:1.2, default = 0.75, show_value = true)) |
| κ | $(@bind ex_kappa Slider(0.0:5.0:60.0, default = 25.0, show_value = true)) |
"""

# ╔═╡ b0000000-0000-0000-0000-000000000010
let
    q = reparam(base_params; beta0 = ex_beta0, kappa = ex_kappa)
    t, Y = simulate(q; method = ex_method, dt = ex_dt, T = 180.0)
    plot_dashboard(
        t,
        Y;
        title = "Interactive — $(method_label(ex_method)), Δt=$(ex_dt), β₀=$(ex_beta0), κ=$(ex_kappa)",
    )
end

# ╔═╡ b0000000-0000-0000-0000-000000000011
let
    q = reparam(base_params; beta0 = ex_beta0, kappa = ex_kappa)
    t, Y = simulate(q; method = ex_method, dt = ex_dt, T = 180.0)
    DataFrame([summarize(t, Y, q)])
end

# ╔═╡ b0000000-0000-0000-0000-000000000012
md"""
# Mission 1 — Decode the Null Fever model

Before the city trusts a simulator, it must understand what is being simulated.
"""

# ╔═╡ b0000000-0000-0000-0000-000000000013
let
    p = base_params
    y0 = initial_state()
    dy0 = rhs(0.0, y0, p)
    DataFrame(
        compartment = collect(string.(COMPARTMENTS)),
        initial_value = y0,
        initial_derivative = dy0,
    )
end

# ╔═╡ b0000000-0000-0000-0000-000000000014
let
    p = base_params
    y0 = initial_state()
    (
        conservation_defect_at_0 = conservation_defect(0.0, y0, p),
        initial_population = sum(y0[POPULATION_IDXS]),
        initial_C = y0[IDX.C],
    )
end

# ╔═╡ b0000000-0000-0000-0000-000000000015
md"""
# Mission 2 — Build the first 180-day forecast

The council asks for a forecast over the 180 days during which Numeria is isolated.

Use explicit Euler with $\Delta t=0.25$ days. Then compare it with a small-step RK4 reference.

### Questions

1. On which day does $I(t)$ reach its maximum?
2. What is the maximum value of $I(t)$?
3. Does the hospital load exceed capacity?
"""

# ╔═╡ b0000000-0000-0000-0000-000000000016
let
    t, Y = simulate(base_params; method = euler_step, dt = 0.25, T = 180.0)
    DataFrame([summarize(t, Y, base_params)])
end

# ╔═╡ b0000000-0000-0000-0000-000000000017
let
    t, Y = simulate(base_params; method = euler_step, dt = 0.25, T = 180.0)
    plot_compartments(t, Y; title = "Null Fever forecast: Euler, Δt=0.25")
end

# ╔═╡ b0000000-0000-0000-0000-000000000018
let
    t, Y = simulate(base_params; method = euler_step, dt = 0.25, T = 180.0)
    plot_dashboard(t, Y; title = "Hospital dashboard: Euler, Δt=0.25")
end

# ╔═╡ b0000000-0000-0000-0000-000000000019
let
    t, Y = simulate(base_params; method = rk4_step, dt = 0.02, T = 180.0)
    DataFrame([summarize(t, Y, base_params)])
end

# ╔═╡ b0000000-0000-0000-0000-00000000001a
let
    t, Y = simulate(base_params; method = rk4_step, dt = 0.02, T = 180.0)
    plot_dashboard(t, Y; title = "Hospital dashboard: RK4 reference")
end

# ╔═╡ b0000000-0000-0000-0000-00000000001b
md"""
# Mission 3 — The simulation can lie

The same model is simulated with different Euler time steps. A larger step is faster, but it may create problems ...

### Question

When does the simulation first become suspicious?
"""

# ╔═╡ b0000000-0000-0000-0000-00000000001c
failure_table = let
    dts = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    rows = [
        merge(
            (dt = dt,),
            summarize(
                simulate(base_params; method = euler_step, dt = dt, T = 180.0)...,
                base_params,
            ),
        ) for dt in dts
    ]
    DataFrame(rows)
end

# ╔═╡ b0000000-0000-0000-0000-00000000001d
let
    plot(
        failure_table.dt,
        failure_table.peak_I;
        marker = :circle,
        xscale = :log10,
        xlabel = "Δt in days",
        ylabel = "Peak I(t)",
        legend = false,
        title = "Euler time step versus peak infectious count",
    )
end

# ╔═╡ b0000000-0000-0000-0000-00000000001e
let
    plt = plot(
        failure_table.dt,
        failure_table.minimum_compartment;
        marker = :circle,
        xscale = :log10,
        xlabel = "Δt in days",
        ylabel = "Minimum of S,V,E,I,Q,R",
        legend = false,
        title = "Euler time step versus smallest compartment value",
    )
    hline!(plt, [0.0]; ls = :dash)
end

# ╔═╡ b0000000-0000-0000-0000-00000000001f
let
    t, Y = simulate(base_params; method = euler_step, dt = 8.0, T = 180.0)
    plot_dashboard(t, Y; title = "Bad forecast: Euler, Δt=8.0")
end

# ╔═╡ b0000000-0000-0000-0000-000000000020
md"""
# Mission 4 — The quarantine shock

Now the city changes only three rates:

```math
\delta=2.5,\qquad \gamma_Q=1.0,\qquad \sigma=1.5.
```

Exposed people become infectious quickly, infectious people are isolated quickly, and isolated people recover quickly. The model now contains faster time scales.

### Questions

1. Which values of $\Delta t$ still work for Euler?
2. Which values create oscillations or negative compartments?
"""

# ╔═╡ b0000000-0000-0000-0000-000000000021
stiff_table = let
    p_stiff = reparam(base_params; delta = 2.5, gamma_q = 1.0, sigma = 1.5)
    stiff_dts = [0.05, 0.1, 0.25, 0.5, 0.6, 0.7, 1.0]
    rows = [
        merge(
            (dt = dt,),
            summarize(simulate(p_stiff; method = euler_step, dt = dt, T = 180.0)..., p_stiff),
        ) for dt in stiff_dts
    ]
    DataFrame(rows)
end

# ╔═╡ b0000000-0000-0000-0000-000000000022
let
    plt = plot(
        stiff_table.dt,
        stiff_table.minimum_compartment;
        marker = :circle,
        xlabel = "Δt in days",
        ylabel = "Minimum of S,V,E,I,Q,R",
        legend = false,
        title = "Stiffness test: Euler minimum compartment",
    )
    hline!(plt, [0.0]; ls = :dash)
end

# ╔═╡ b0000000-0000-0000-0000-000000000023
md"""
# Mission 5 — Three engines for the same city: Euler, Heun, RK4

Euler is simple. Heun is a second-order predictor-corrector method. RK4 is a fourth-order method.

We compare all three against a small-step RK4 reference.

### Questions

1. Which method is most accurate for the same time step?
2. Does a higher-order method completely remove stability problems?
3. Why might your team still use simple methods in large simulations?
"""

# ╔═╡ b0000000-0000-0000-0000-000000000024
method_table = let
    t_ref, Y_ref = simulate(base_params; method = rk4_step, dt = 0.01, T = 180.0)
    compare_dts = [0.25, 0.5, 1.0, 2.0]
    rows = NamedTuple[]
    for step in (euler_step, heun_step, rk4_step), dt in compare_dts
        t, Y = simulate(base_params; method = step, dt = dt, T = 180.0)
        push!(
            rows,
            (
                method = method_label(step),
                dt = dt,
                error_in_I = error_against_reference(t, Y, t_ref, Y_ref, :I),
                minimum_compartment = summarize(t, Y, base_params).minimum_compartment,
            ),
        )
    end
    DataFrame(rows)
end

# ╔═╡ b0000000-0000-0000-0000-000000000025
let
    plt = plot(;
        xscale = :log10,
        yscale = :log10,
        xlabel = "Δt in days",
        ylabel = "max |I_method - I_reference|",
        legend = :topleft,
        title = "Method comparison: error in infectious curve",
    )
    for method in ["Euler", "Heun", "RK4"]
        data = method_table[method_table.method .== method, :]
        plot!(plt, data.dt, data.error_in_I; marker = :circle, label = method)
    end
    plt
end

# ╔═╡ b0000000-0000-0000-0000-000000000026
md"""
# Mission 6 — Measure convergence order

A numerical method has order $p$ if the error behaves approximately like

```math
\text{error}(\Delta t)\approx C(\Delta t)^p.
```

On a log-log plot, this looks like a straight line of slope $p$.

If two time steps $\Delta t_1$ and $\Delta t_2$ give errors $e_1$ and $e_2$, the order can be estimated by

```math
p\approx \frac{\log(e_1/e_2)}{\log(\Delta t_1/\Delta t_2)}.
```

### Question

What observed order do you get for each scheme?
"""

# ╔═╡ b0000000-0000-0000-0000-000000000027
conv_table = let
    t_ref, Y_ref = simulate(base_params; method = rk4_step, dt = 0.01, T = 180.0)
    conv_dts = [1.6, 0.8, 0.4, 0.2, 0.1]
    rows = NamedTuple[]
    for step in (euler_step, heun_step, rk4_step), dt in conv_dts
        t, Y = simulate(base_params; method = step, dt = dt, T = 180.0)
        push!(
            rows,
            (
                method = method_label(step),
                dt = dt,
                error_in_I = error_against_reference(t, Y, t_ref, Y_ref, :I),
            ),
        )
    end
    DataFrame(rows)
end

# ╔═╡ b0000000-0000-0000-0000-000000000028
let
    plt = plot(;
        xscale = :log10,
        yscale = :log10,
        xlabel = "Δt in days",
        ylabel = "max |I_method - I_reference|",
        legend = :topleft,
        title = "Convergence plot",
    )
    for method in ["Euler", "Heun", "RK4"]
        data = conv_table[conv_table.method .== method, :]
        plot!(plt, data.dt, data.error_in_I; marker = :circle, label = method)
    end
    plt
end

# ╔═╡ b0000000-0000-0000-0000-000000000029
let
    orders = NamedTuple[]
    for method in ["Euler", "Heun", "RK4"]
        data = sort(conv_table[conv_table.method .== method, :], :dt; rev = true)
        dts_arr = data.dt
        errs = data.error_in_I
        for j = 1:(length(dts_arr)-1)
            ord = log(errs[j] / errs[j+1]) / log(dts_arr[j] / dts_arr[j+1])
            push!(
                orders,
                (
                    method = method,
                    from_dt = dts_arr[j],
                    to_dt = dts_arr[j+1],
                    observed_order = ord,
                ),
            )
        end
    end
    DataFrame(orders)
end

# ╔═╡ b0000000-0000-0000-0000-00000000002a
md"""
# Mission 7 — Intervention cards

The city has limited resources. Each policy card has a cost. The council can spend at most **5 points**.

The goal is not only to reduce infections, but also to avoid hospital overload.

### Questions

1. Which policy combination gives the smallest hospital overload?
2. Which gives the smallest cumulative number of infections?
3. Is the best policy obvious, or is there a tradeoff?
4. Can several weak interventions combine into a strong one?
"""

# ╔═╡ b0000000-0000-0000-0000-00000000002b
cards = policy_cards()

# ╔═╡ b0000000-0000-0000-0000-00000000002c
policy_table = let
    feasible = NamedTuple[]
    for r = 0:length(cards)
        for chosen in (r == 0 ? [eltype(cards)[]] : combinations(cards, r))
            q, names, cost = apply_policy_set(base_params, chosen)
            cost <= 5 || continue
            t, Y = simulate(q; method = rk4_step, dt = 0.25, T = 180.0)
            s = summarize(t, Y, q)
            push!(
                feasible,
                (
                    policy = isempty(names) ? "no action" : join(names, " + "),
                    cost = cost,
                    peak_hospital_load = s.peak_hospital_load,
                    overload_days = s.overload_days,
                    final_cumulative_cases = s.final_cumulative_cases,
                    peak_I = s.peak_I,
                ),
            )
        end
    end
    sort(
        DataFrame(feasible),
        [:overload_days, :peak_hospital_load, :final_cumulative_cases],
    )
end

# ╔═╡ b0000000-0000-0000-0000-00000000002d
let
    plt = scatter(
        policy_table.final_cumulative_cases,
        policy_table.peak_hospital_load;
        label = "feasible policies",
        xlabel = "Final cumulative cases",
        ylabel = "Peak hospital load",
        legend = :topright,
        title = "Policy tradeoff: all feasible policies",
    )
    hline!(plt, [600.0]; ls = :dash, label = "hospital capacity")
end

# ╔═╡ b0000000-0000-0000-0000-00000000002e
md"""
# Mission 8 — The hidden outbreak parameters

The council receives noisy weekly reports of infectious cases. Two parameters are uncertain: $\beta_0$ and $\kappa$.

Here $\beta_0$ controls baseline transmission, while $\kappa$ controls behavioral feedback.

The task is to search over possible values and find the pair that best explains the data.

### Questions

1. Where is the minimum of the surface?
2. What would a broad valley mean for policy decisions?
3. Can two different parameter pairs produce similar epidemic curves?

> Note: the synthetic "observed" data uses Julia's RNG, so the exact noise differs from the Python notebook, but the recovered parameters still land near $\beta_0=0.78,\ \kappa=18$.
"""

# ╔═╡ b0000000-0000-0000-0000-00000000002f
weeks, observed_I, true_params = make_weekly_observations(base_params)

# ╔═╡ b0000000-0000-0000-0000-000000000030
let
    scatter(
        weeks,
        observed_I;
        label = "weekly reports",
        xlabel = "Day",
        ylabel = "Reported infectious cases",
        legend = :topright,
        title = "Noisy weekly reports",
    )
end

# ╔═╡ b0000000-0000-0000-0000-000000000031
beta_grid, kappa_grid, loss_surface, best_beta0, best_kappa, best_loss = let
    bg = collect(range(0.60, 0.92; length = 25))
    kg = collect(range(0.0, 50.0; length = 25))
    L = [calibration_loss(b, k, weeks, observed_I, base_params) for k in kg, b in bg]
    idx = argmin(L)
    (bg, kg, L, bg[idx[2]], kg[idx[1]], L[idx])
end

# ╔═╡ b0000000-0000-0000-0000-000000000032
(
    best_beta0 = round(best_beta0; digits = 4),
    best_kappa = round(best_kappa; digits = 4),
    best_loss = round(best_loss; digits = 2),
    hidden_beta0 = true_params.beta0,
    hidden_kappa = true_params.kappa,
)

# ╔═╡ b0000000-0000-0000-0000-000000000033
let
    plt = surface(
        beta_grid,
        kappa_grid,
        loss_surface;
        xlabel = "beta0",
        ylabel = "kappa",
        zlabel = "mean squared error",
        title = "Calibration loss surface",
        camera = (-160, 25),
    )
    scatter!(plt, [best_beta0], [best_kappa], [best_loss]; label = "best fit", ms = 5)
end

# ╔═╡ b0000000-0000-0000-0000-000000000034
let
    p_fit = reparam(base_params; beta0 = best_beta0, kappa = best_kappa)
    t_fit, Y_fit = simulate(p_fit; method = rk4_step, dt = 0.1, T = 180.0)
    fit_I_weekly = interp_component(t_fit, Y_fit, weeks, :I)
    plt = scatter(
        weeks,
        observed_I;
        label = "weekly reports",
        xlabel = "Day",
        ylabel = "Infectious cases",
        legend = :topright,
        title = "Observed data versus fitted curve",
    )
    plot!(plt, weeks, fit_I_weekly; marker = :circle, label = "best fitted model")
end

# ╔═╡ b0000000-0000-0000-0000-000000000035
md"""
# Mission 9 — Harder mission: robust policy under uncertainty

The fitted parameters are not exact. The city must choose a policy that works not only for the best-fit scenario, but also for nearby scenarios.

We test each policy under a small uncertainty box around the fitted parameters:

```math
\beta_0 \in \{0.95\widehat\beta_0,\ \widehat\beta_0,\ 1.05\widehat\beta_0\},\qquad
\kappa \in \{\widehat\kappa-10,\ \widehat\kappa,\ \widehat\kappa+10\}.
```

### Questions

1. Which policy has the smallest worst-case overload?
2. Is it the same as the best policy from Mission 7?
3. Why might a robust policy be preferable to a policy that only works for one fitted model?
"""

# ╔═╡ b0000000-0000-0000-0000-000000000036
robust_table = let
    scenarios = [
        reparam(base_params; beta0 = best_beta0 * bf, kappa = max(0.0, best_kappa + ks)) for bf in (0.95, 1.0, 1.05) for ks in (-10.0, 0.0, 10.0)
    ]
    rows = NamedTuple[]
    for r = 0:length(cards)
        for chosen in (r == 0 ? [eltype(cards)[]] : combinations(cards, r))
            _, names, cost = apply_policy_set(base_params, chosen)
            cost <= 5 || continue
            overloads = Float64[]
            peaks = Float64[]
            cumulative = Float64[]
            for scenario in scenarios
                q_scenario, _, _ = apply_policy_set(scenario, chosen)
                t, Y = simulate(q_scenario; method = rk4_step, dt = 0.5, T = 180.0)
                s = summarize(t, Y, q_scenario)
                push!(overloads, s.overload_days)
                push!(peaks, s.peak_hospital_load)
                push!(cumulative, s.final_cumulative_cases)
            end
            push!(
                rows,
                (
                    policy = isempty(names) ? "no action" : join(names, " + "),
                    cost = cost,
                    worst_overload_days = maximum(overloads),
                    worst_peak_hospital_load = maximum(peaks),
                    worst_cumulative_cases = maximum(cumulative),
                ),
            )
        end
    end
    sort(
        DataFrame(rows),
        [:worst_overload_days, :worst_peak_hospital_load, :worst_cumulative_cases],
    )
end

# ╔═╡ b0000000-0000-0000-0000-000000000037
let
    plt = scatter(
        robust_table.worst_cumulative_cases,
        robust_table.worst_peak_hospital_load;
        label = "feasible policies",
        xlabel = "Worst-case cumulative cases",
        ylabel = "Worst-case peak hospital load",
        legend = :topright,
        title = "Robust policy tradeoff: all feasible policies",
    )
    hline!(plt, [600.0]; ls = :dash, label = "hospital capacity")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Combinatorics = "~1.1.0"
DataFrames = "~1.8.2"
Plots = "~1.41.6"
PlutoUI = "~0.7.83"
StaticArrays = "~1.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "0d772f38e365387f829c4c399b7d2aaab9c3d9cf"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "c761b00e7755700f9cdf5b02039939d1359330e1"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.1.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f954322d5de03ec630d177cda203dcd92b6be399"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.26"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "6fada551286ab6ea4ca1628cb2de9f166a2ec966"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.26+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─b0000000-0000-0000-0000-000000000001
# ╟─b0000000-0000-0000-0000-000000000002
# ╠═b0000000-0000-0000-0000-000000000003
# ╟─b0000000-0000-0000-0000-000000000004
# ╠═b0000000-0000-0000-0000-000000000005
# ╠═b0000000-0000-0000-0000-000000000006
# ╠═b0000000-0000-0000-0000-000000000007
# ╠═b0000000-0000-0000-0000-000000000008
# ╠═b0000000-0000-0000-0000-000000000009
# ╠═b0000000-0000-0000-0000-00000000000a
# ╠═b0000000-0000-0000-0000-00000000000b
# ╠═b0000000-0000-0000-0000-00000000000c
# ╠═b0000000-0000-0000-0000-00000000000d
# ╠═b0000000-0000-0000-0000-00000000000e
# ╟─b0000000-0000-0000-0000-00000000000f
# ╠═b0000000-0000-0000-0000-000000000010
# ╠═b0000000-0000-0000-0000-000000000011
# ╟─b0000000-0000-0000-0000-000000000012
# ╠═b0000000-0000-0000-0000-000000000013
# ╠═b0000000-0000-0000-0000-000000000014
# ╟─b0000000-0000-0000-0000-000000000015
# ╠═b0000000-0000-0000-0000-000000000016
# ╠═b0000000-0000-0000-0000-000000000017
# ╠═b0000000-0000-0000-0000-000000000018
# ╠═b0000000-0000-0000-0000-000000000019
# ╠═b0000000-0000-0000-0000-00000000001a
# ╟─b0000000-0000-0000-0000-00000000001b
# ╠═b0000000-0000-0000-0000-00000000001c
# ╠═b0000000-0000-0000-0000-00000000001d
# ╠═b0000000-0000-0000-0000-00000000001e
# ╠═b0000000-0000-0000-0000-00000000001f
# ╟─b0000000-0000-0000-0000-000000000020
# ╠═b0000000-0000-0000-0000-000000000021
# ╠═b0000000-0000-0000-0000-000000000022
# ╟─b0000000-0000-0000-0000-000000000023
# ╠═b0000000-0000-0000-0000-000000000024
# ╠═b0000000-0000-0000-0000-000000000025
# ╟─b0000000-0000-0000-0000-000000000026
# ╠═b0000000-0000-0000-0000-000000000027
# ╠═b0000000-0000-0000-0000-000000000028
# ╠═b0000000-0000-0000-0000-000000000029
# ╟─b0000000-0000-0000-0000-00000000002a
# ╠═b0000000-0000-0000-0000-00000000002b
# ╠═b0000000-0000-0000-0000-00000000002c
# ╠═b0000000-0000-0000-0000-00000000002d
# ╟─b0000000-0000-0000-0000-00000000002e
# ╠═b0000000-0000-0000-0000-00000000002f
# ╠═b0000000-0000-0000-0000-000000000030
# ╠═b0000000-0000-0000-0000-000000000031
# ╠═b0000000-0000-0000-0000-000000000032
# ╠═b0000000-0000-0000-0000-000000000033
# ╠═b0000000-0000-0000-0000-000000000034
# ╟─b0000000-0000-0000-0000-000000000035
# ╠═b0000000-0000-0000-0000-000000000036
# ╠═b0000000-0000-0000-0000-000000000037
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
