"""
every-visit MC algo (pg.19 Lecture 3: Value-based methods slide ),
simpler than first-visit MC.
params:
    - states, unique states
    - S, s0, s1,...
    - R, a "matrix" of transition rewards ∈ Float64
    - γ, discount factor ∈ (0,1)
returns:
    - similar(S)
"""
function evmc(states, S, R, γ)
    n = zeros(Int, length(states)) # n_ts[i] := counter of visiting state i
    Vs = zeros(length(states)) # output
    for i ∈ eachindex(S) # foreach episode
        G = 0. # reset G foreach eps
        # compute the return G backwards:
        println("episode",i)
        for t ∈ reverse(eachindex(S[i]))
            G = γ*G + R[i][t] # discount G_{t+1}
            n[S[i][t]] += 1 # update counter
            Vs[S[i][t]] += G # update return of S_t
            println([t, S[i][t], n[S[i][t]], G])
        end
    end
    display(n)
    display(Vs)
    return Vs ./ n # average V(s) := V(s)/n(s)
end


function main()
    # encode {"A B T"} = {1 2 3}, {"stay switch"} = {1 2}, like  a graph struct
    eps = [[1,1,2,3], [1,2,1,2,3], [1,2,3]]
    R = Dict() # "matrix" of reward for state transitions
    R[1, 1] = 1.
    R[1, 2] = 0.
    R[2, 1] = 1.
    R[2, 3] = 2.
    # or transform reward matrix to reward sequence given episodes (or just manually hardcode, easier tbh):
    R_eps = Vector{Vector{Float64}}()
    for k ∈ eachindex(eps)
        temp = Vector{Float64}()
        for i ∈ eachindex(eps[k][1:end-1])
            push!(temp, R[eps[k][i], eps[k][i+1]])
        end
        push!(R_eps, temp)
    end
    # or list of lists of where each list := one episode:
    S = [[1,1,2],[1,2,1,2], [1,2]]
    A = [[1,2,1],[2,2,2,1], [2,1]] # actions dont matter for computation
    R = Vector{Vector{Float64}}([[1,0,2], [0,1,0,2], [0,2]])
    # or list of dicts...:
    trjs = [Dict("s"=>1, "a"=>1, "r"=>1.)]
    # initialize learning rates:
    γ = .5; α = .1
    Vs = evmc(["A", "B"], S, R, γ)
    display(Vs)
end