# Imports ---------------------------------------------------------------------

from collections import OrderedDict
import pulp
from pulp import LpVariable as Var
from pulp import lpSum


# Functions -------------------------------------------------------------------

# One-sided market problem - questions 1 and 3
def function_1(data, output):
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Variables <to be completed by the student>
    gen = [Var(f'gen_{q}') for q in range(len(data.get("gen_price")))]

    # Objective function <to be completed by the student>
    m += lpSum(gen[n] * data.get("gen_price")[n] for n in range(len(gen)))

    # Constraints <to be completed by the student>
    c1 = lpSum(q for q in gen) == 150
    m += c1

    for n in range(len(gen)):
        m += (data.get("gen_quantity")[n] >= gen[n])
        m += gen[n] >= 0

    # Solving
    # m.solve(pulp.PULP_CBC_CMD(msg=True))
    m.solve()

    # Solution
    output["objective_function"] = m.objective.value()
    output["LP status"].append(pulp.LpStatus[m.status])

    output["marginal_price"] = c1.pi  # <to be replaced by the student>
    for i in range(len(data["gen_quantity"])):
        output["gen_quantity"][i] = gen[i].value()  # <to be replaced by the student>

    return output


# Two-sided market problem - questions 2, 3 and 4
def function_2(data, output):
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Variables for question 2 <to be completed by the student>
    gen = [Var(f'gen_{q}') for q in range(len(data.get("gen_quantity")))]
    load = [Var(f'load_{l}') for l in range(len(data.get("load_quantity")))]

    # Variables only for question 4 <to be completed by the student>
    theta = [Var(f'theta_{n}') for n in data.get("buses")]
    # Objective function <to be completed by the student>
    m += (lpSum(gen[n] * data.get("gen_price")[n] for n in range(len(gen)))
          - lpSum(load[n] * data.get("load_price")[n] for n in range(len(load)))
          )

    # Constraints for question 2  <to be completed by the student>
    c1 = lpSum(g for g in gen) == lpSum(l for l in load)
    m += c1

    for n in range(len(gen)):
        m += (data.get("gen_quantity")[n] >= gen[n])
        m += gen[n] >= 0
    for n in range(len(load)):
        m += (data.get("load_quantity")[n] >= load[n])
        m += load[n] >= 0

    # Constraints only for question 4 <to be completed by the student>
    for n in data.get("buses"):
        gen_total = 0
        load_total = 0
        for g in data.get("gen network location")[n]:
            gen_total += gen[g]
        for l in data.get("load network location")[n]:
            load_total += load[l]
        sum_branches = 0
        for bus_from_index in range(len(data.get("from bus"))):
            if data.get("from bus")[bus_from_index] == n:
                bus_to_go = data.get("to bus")[bus_from_index]
                reactance = data.get("reactance")[bus_from_index]
                branch_limit = data.get("branch limit")[bus_from_index]

                branch = (theta[n] - theta[bus_to_go])/reactance
                m += branch <= branch_limit
                m += -branch_limit <= branch
                sum_branches += branch
        m += gen_total - load_total == sum_branches

    # Solving
    m.solve(pulp.PULP_CBC_CMD(msg=True))

    # Solution
    output["objective_function"] = m.objective.value()
    output["LP status"].append(pulp.LpStatus[m.status])

    output["marginal_price"] = c1.pi  # <to be replaced by the student>
    for i in range(len(data["gen_quantity"])):
        output["gen_quantity"][i] = gen[i].value()  # <to be replaced by the student>
    for i in range(len(data["load_quantity"])):
        output["load_quantity"][i] = load[i].value()  # <to be replaced by the student>
    for l in range(len(data["from bus"])):
        bus_from = data["from bus"][l]
        bus_to_go = data["to bus"][l]
        reactance = data.get("reactance")[l]
        output["branch power flow"][l] = (theta[bus_from].value() - theta[bus_to_go].value())/reactance  # <to be replaced by the student> - only for question 4

    return output


# Main ------------------------------------------------------------------------

if __name__ == "__main__":
    # Input data
    data = OrderedDict()

    # Total System load - One-sided market
    data["total_load"] = 150

    # Table 1 data
    data["gen_price"] = [39.0, 35.0, 29.0, 58.0, 9.0, 9.0]
    data["gen_quantity"] = [60.0, 50.0, 20.0, 80.0, 10.0, 10.0]

    # Table 2 data
    data["load_price"] = [40.0, 45.0, 58.0, 55.0, 48.0, 35.0]
    data["load_quantity"] = [27.0, 30.0, 24.0, 45.0, 30.0, 24.0]

    # Table 3 data
    data["from bus"] = [0, 0, 1, 1, 2, 3, 4]
    data["to bus"] = [1, 2, 2, 4, 3, 4, 5]
    data["reactance"] = [0.06, 0.24, 0.18, 0.12, 0.03, 0.24, 0.03]
    data["branch limit"] = [40, 40, 40, 40, 40, 40, 40]

    # Figure 1 data
    # Location of generators in the network, e.g. the bus 0 has generators 0 and 1
    data["gen network location"] = [[0, 1], [2, 3], [], [], [4], [5]]
    # Location of loads in the network, e.g. the bus 1 has load 0
    data["load network location"] = [[], [0], [1, 2], [3], [4, 5], []]
    # buses
    data["buses"] = [0, 1, 2, 3, 4, 5]

    # Output data to be printed
    output = OrderedDict()

    output["gen_quantity"] = [0, 0, 0, 0, 0, 0]  # Accepted generation quantities
    output["load_quantity"] = [0, 0, 0, 0, 0, 0]  # Accepted load quantities
    output["marginal_price"] = 0.0  # Marginal price
    output["objective_function"] = 0.0
    output["branch power flow"] = [0, 0, 0, 0, 0, 0, 0]
    output["LP status"] = []  # Saves LP status

    # Run one-sided market (questions 1 and 3)
    # output = function_1(data, output)

    # Run two-side market (questions 2, 3, and 4)
    output = function_2(data, output)

    # Print outputs

    print("LP Status:", output["LP status"][0])
    print("Accepted generation quantities (MW):", output["gen_quantity"])
    print("Accepted load quantities (MW):", output["load_quantity"])  # This is not required for the one-sided market.
    print("Branch power flow (MW):", output["branch power flow"])  # This is not required for question 4.
    print("Marginal price ($/MW):", round(output["marginal_price"], 2))
    print("Objective function ($):", round(output["objective_function"], 2))
