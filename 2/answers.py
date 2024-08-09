# Imports ---------------------------------------------------------------------

from collections import OrderedDict

import numpy as np
import pulp
from pulp import LpVariable as Var, LpVariable
from pulp import lpSum


# Functions -------------------------------------------------------------------

# Question 1 [15 marks]
def function_1(data, output):
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Indices:
    time = [*range(0, 24, 1)]
    ev = data.get("EV time")

    # Parameters:
    appsCost = data.get("appiances")
    solarGen = data.get("solar")
    buyP = data.get("buying price")
    sellP = data.get("selling price")

    # Variables:
    # Is buying/selling at each t
    VisBuy = {s: Var('VisBuy_{}'.format(s), cat='Binary') for s in time}
    VisSell = {s: Var('VisSell_{}'.format(s), cat='Binary') for s in time}
    VisBuySellMid = {s: Var('VisMid_{}'.format(s), cat='Binary') for s in time}
    # how much to VBuy/VSell at each t
    VBuy = {q: Var(f'VBuy_{q}', lowBound=0) for q in time}
    VSell = {q: Var(f'VSell_{q}', lowBound=0) for q in time}
    VBuyActual = {q: Var(f'VBuyActual_{q}', lowBound=0) for q in time}
    VSellActual = {q: Var(f'VSellActual_{q}', lowBound=0) for q in time}

    # SOC for vec at each t
    VSoc = {q: Var(f'VSoc_{q}', lowBound=0, upBound=39) for q in time}
    # Solar generation at each t
    VSolar = {q: Var(f'VSolar_{q}', lowBound=0) for q in time}

    # Is charging/discharging at each ev
    VisC = {s: Var('VCharge_{}'.format(s), cat='Binary') for s in ev}
    VisD = {s: Var('VDischarge_{}'.format(s), cat='Binary') for s in ev}
    VisCDMid = {s: Var('VDisCDMid_{}'.format(s), cat='Binary') for s in ev}
    # Hot much to charge/discharge at each t
    VptPlus = {q: Var(f'VptPlus_{q}', lowBound=0, upBound=3.7) for q in time}
    VptMinus = {q: Var(f'VptMinus_{q}', lowBound=0, upBound=3.7) for q in time}
    VptPlusActual = {q: Var(f'VptPlusActual_{q}', lowBound=0, upBound=3.7) for q in time}
    VptMinusActual = {q: Var(f'VptMinusActual_{q}', lowBound=0, upBound=3.7) for q in time}

    # Objective function
    m += lpSum((VBuyActual[t] * buyP[t]) for t in time) - lpSum((VSellActual[t] * sellP[t]) for t in time)

    # Constraints
    M = 50
    for t in time:
        # if VisBuy[t]/VisSell[t] is 0, then VBuyActual[t]/VSellActual[t] must be 0
        # VisBuy[t] and VisSell[t] can be 0 at same time, cannot be 1 at same time.
        m += VisBuy[t] <= 1 - VisBuySellMid[t]
        m += VisBuy[t] <= 1 - VisSell[t]
        m += VisBuy[t] >= 1 - VisBuySellMid[t] + 1 - VisSell[t] - 1

        m += VBuyActual[t] <= M * VisBuy[t]
        m += VBuyActual[t] <= VBuy[t]
        m += VBuy[t] - VBuyActual[t] <= M * (1 - VisBuy[t])

        m += VSellActual[t] <= M * VisSell[t]
        m += VSellActual[t] <= VSell[t]
        m += VSell[t] - VSellActual[t] <= M * (1 - VisSell[t])

        # ele into house == ele out house(consume)
        m += appsCost[t] + VptPlusActual[t] + VSellActual[t] == VSolar[t] + VptMinusActual[t] + VBuyActual[t]

        # ele into house fit appliances
        m += VSolar[t] + VptMinusActual[t] + VBuyActual[t] >= appsCost[t]

        # solar generation upBound
        m += VSolar[t] <= solarGen[t]

        if t not in ev:
            m += VptPlus[t] == 0
            m += VptMinus[t] == 0
            m += VptPlusActual[t] == 0
            m += VptMinusActual[t] == 0
            if t != 10 and t != 20:
                m += VSoc[t] == 0

    m += VSoc[ev[0]] == 30
    for t in ev:
        # if VisC[t]/VisD[t] is 0, then VptPlusActual[t]/VptMinusActual[t] must be 0
        # VisC[t] and VisD[t] can be 0 at same time, cannot be 1 at same time.
        m += VisC[t] <= 1 - VisCDMid[t]
        m += VisC[t] <= 1 - VisD[t]
        m += VisC[t] >= 1 - VisCDMid[t] + 1 - VisD[t] - 1

        m += VptPlusActual[t] <= M * VisC[t]
        m += VptPlusActual[t] <= VptPlus[t]
        m += VptPlus[t] - VptPlusActual[t] <= M * (1 - VisC[t])

        m += VptMinusActual[t] <= M * VisD[t]
        m += VptMinusActual[t] <= VptMinus[t]
        m += VptMinus[t] - VptMinusActual[t] <= M * (1 - VisD[t])

        m += VSoc[t + 1] == VSoc[t] + (0.9 * VptPlusActual[t]) - (VptMinusActual[t] * 10 / 9)

    m += VSoc[ev[-1] + 1] >= 33

    # Solving
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    Cost = 0
    vbuy = []
    vsell = []
    for t in time:
        vbuy.append(VBuyActual[t].value())
        vsell.append(VSellActual[t].value())
        Cost = Cost + (VisBuy[t].value() * VBuyActual[t].value() * buyP[t]) - (
                    VisSell[t].value() * VSellActual[t].value() * sellP[t])

    # Solution
    output["MILP status"].append(pulp.LpStatus[m.status])

    # Round the results to 1 decimal

    # Electricity cost of the smart home (in $)
    output["electricity cost"] = round(Cost, 1)
    # Electricity bought (in kWh)
    output["electricity bought"] = round(sum(vbuy), 1)
    # Electricity sold (in kWh)
    output["electricity sold"] = round(sum(vsell), 1)

    for t in range(len(data["solar"])):
        # Solar generation (in kW)
        output["solar"][t] = round(VSolar[t].value(), 1)
        # State-of-charge (in kWh)
        output["SOC"][t] = round(VSoc[t].value(), 1)
        # Battery charging (in kW)
        output["Charging"][t] = round(VptPlus[t].value(), 1)
        # Battery discharging (in kW)
        output["Discharging"][t] = round(VptMinus[t].value(), 1)

    return output


# Question 2 [15 marks]
def function_2(data, output):
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Indices:
    time = [*range(0, 24, 1)]
    ev = data.get("EV time")
    x = [*range(1, 23, 1)]

    # Parameters:
    appsCost = data.get("appiances")
    solarGen = data.get("solar")
    buyP = data.get("buying price")
    sellP = data.get("selling price")

    # Variables:
    # Is buying/selling at each t
    VisBuy = {s: Var('VisBuy_{}'.format(s), cat='Binary') for s in time}
    VisSell = {s: Var('VisSell_{}'.format(s), cat='Binary') for s in time}
    VisBuySellMid = {s: Var('VisMid_{}'.format(s), cat='Binary') for s in time}
    # how much to VBuy/VSell at each t
    VBuy = {q: Var(f'VBuy_{q}', lowBound=0) for q in time}
    VSell = {q: Var(f'VSell_{q}', lowBound=0) for q in time}
    VBuyActual = {q: Var(f'VBuyActual_{q}', lowBound=0) for q in time}
    VSellActual = {q: Var(f'VSellActual_{q}', lowBound=0) for q in time}

    # SOC for vec at each t
    VSoc = {q: Var(f'VSoc_{q}', lowBound=0, upBound=39) for q in time}
    # Solar generation at each t
    VSolar = {q: Var(f'VSolar_{q}', lowBound=0) for q in time}

    # Is charging/discharging at each ev
    VisC = {s: Var('VCharge_{}'.format(s), cat='Binary') for s in ev}
    VisD = {s: Var('VDischarge_{}'.format(s), cat='Binary') for s in ev}
    VisCDMid = {s: Var('VDisCDMid_{}'.format(s), cat='Binary') for s in ev}
    # Hot much to charge/discharge at each t
    VptPlus = {q: Var(f'VptPlus_{q}', lowBound=0, upBound=3.7) for q in time}
    VptMinus = {q: Var(f'VptMinus_{q}', lowBound=0, upBound=3.7) for q in time}
    VptPlusActual = {q: Var(f'VptPlusActual_{q}', lowBound=0, upBound=3.7) for q in time}
    VptMinusActual = {q: Var(f'VptMinusActual_{q}', lowBound=0, upBound=3.7) for q in time}

    VNewApps = {s: Var('VNewApps_{}'.format(s), cat='Binary') for s in x}
    VNewApp1 = {s: Var('VNewApp1_{}'.format(s), cat='Binary') for s in time}
    VNewApp2 = {s: Var('VNewApp2_{}'.format(s), cat='Binary') for s in time}
    VNewApp3 = {s: Var('VNewApp3_{}'.format(s), cat='Binary') for s in time}

    VNewAppSum = {s: Var('VNewAppSum_{}'.format(s), lowBound=0) for s in time}

    # Objective function
    m += lpSum((VBuyActual[t] * buyP[t]) for t in time) - lpSum((VSellActual[t] * sellP[t]) for t in time)

    # Constraints
    m += lpSum(VNewApps[t] for t in x) == 1
    m += lpSum(VNewApp1[t] for t in time) == 1
    m += lpSum(VNewApp2[t] for t in time) == 1
    m += lpSum(VNewApp3[t] for t in time) == 1
    m += lpSum(VNewAppSum[t] for t in time) == 3

    M = 50
    for t in x:
        m += VNewApp1[t] == VNewApps[t]
        m += VNewApp2[t + 1] == VNewApps[t]
        m += VNewApp3[t - 1] == VNewApps[t]

    for t in time:
        # join VNewApp1,VNewApp2,VNewApp3 as VNewAppSum
        m += VNewAppSum[t] >= VNewApp1[t]
        m += VNewAppSum[t] >= VNewApp2[t]
        m += VNewAppSum[t] >= VNewApp3[t]

        m += VNewAppSum[t] <= VNewApp1[t] + VNewApp2[t] + VNewApp3[t]

        # if VisBuy[t]/VisSell[t] is 0, then VBuyActual[t]/VSellActual[t] must be 0
        # VisBuy[t] and VisSell[t] can be 0 at same time, cannot be 1 at same time.
        m += VisBuy[t] <= 1 - VisBuySellMid[t]
        m += VisBuy[t] <= 1 - VisSell[t]
        m += VisBuy[t] >= 1 - VisBuySellMid[t] + 1 - VisSell[t] - 1

        m += VBuyActual[t] <= M * VisBuy[t]
        m += VBuyActual[t] <= VBuy[t]
        m += VBuy[t] - VBuyActual[t] <= M * (1 - VisBuy[t])

        m += VSellActual[t] <= M * VisSell[t]
        m += VSellActual[t] <= VSell[t]
        m += VSell[t] - VSellActual[t] <= M * (1 - VisSell[t])

        # ele into house == ele out house(consume)
        m += appsCost[t] + VptPlusActual[t] + VSellActual[t] + VNewAppSum[t] == VSolar[t] + VptMinusActual[t] + \
             VBuyActual[t]

        # # ele into house fit appliances
        m += VSolar[t] + VptMinusActual[t] + VBuyActual[t] >= appsCost[t] + VNewAppSum[t]

        # solar generation upBound
        m += VSolar[t] <= solarGen[t]

        if t not in ev:
            m += VptPlus[t] == 0
            m += VptMinus[t] == 0
            m += VptPlusActual[t] == 0
            m += VptMinusActual[t] == 0
            if t != 10 and t != 20:
                m += VSoc[t] == 0

    m += VSoc[ev[0]] == 30
    for t in ev:
        # if VisC[t]/VisD[t] is 0, then VptPlusActual[t]/VptMinusActual[t] must be 0
        # VisC[t] and VisD[t] can be 0 at same time, cannot be 1 at same time.
        m += VisC[t] <= 1 - VisCDMid[t]
        m += VisC[t] <= 1 - VisD[t]
        m += VisC[t] >= 1 - VisCDMid[t] + 1 - VisD[t] - 1

        m += VptPlusActual[t] <= M * VisC[t]
        m += VptPlusActual[t] <= VptPlus[t]
        m += VptPlus[t] - VptPlusActual[t] <= M * (1 - VisC[t])

        m += VptMinusActual[t] <= M * VisD[t]
        m += VptMinusActual[t] <= VptMinus[t]
        m += VptMinus[t] - VptMinusActual[t] <= M * (1 - VisD[t])

        m += VSoc[t + 1] == VSoc[t] + (0.9 * VptPlusActual[t]) - (VptMinusActual[t] * 10 / 9)

    m += VSoc[ev[-1] + 1] >= 33

    # Solving
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    Cost = 0
    NeAappliance = []
    vbuy = []
    vsell = []
    for t in time:
        vbuy.append(VBuyActual[t].value())
        vsell.append(VSellActual[t].value())
        if VNewAppSum[t].value() == 1:
            NeAappliance.append(t)
        Cost = Cost + (VisBuy[t].value() * VBuyActual[t].value() * buyP[t]) - (
                    VisSell[t].value() * VSellActual[t].value() * sellP[t])

    # Solving
    m.solve()
    # m.solve(pulp.PULP_CBC_CMD(msg=0))

    # Solution
    output["MILP status"].append(pulp.LpStatus[m.status])

    # Round the results to 1 decimal

    # Electricity cost of the smart home (in $)
    output["electricity cost"] = round(Cost, 1)
    # Electricity bought (in kWh)
    output["electricity bought"] = round(sum(vbuy), 1)
    # Electricity sold (in kWh)
    output["electricity sold"] = round(sum(vsell), 1)
    # New appliance (operating hours)
    output["new appliance"] = NeAappliance

    for t in range(len(data["solar"])):
        # Solar generation (in kW)
        output["solar"][t] = round(VSolar[t].value(), 1)
        # State-of-charge (in kWh)
        output["SOC"][t] = round(VSoc[t].value(), 1)
        # Battery charging (in kW)
        output["Charging"][t] = round(VptPlus[t].value(), 1)
        # Battery discharging (in kW)
        output["Discharging"][t] = round(VptMinus[t].value(), 1)

    return output


# Question 3 [15 marks]
def function_3(data, output):
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Indices:
    time = [*range(0, 24, 1)]
    ev = data.get("EV time")
    netfee = [0.5, 0.5, 1, 1, 1.5, 1.5]
    points_n = [0, 4, 4, 8, 8, 1e5]

    # Parameters:
    appsCost = data.get("appiances")
    solarGen = data.get("solar")
    buyP = data.get("buying price")
    sellP = data.get("selling price")

    # Variables:
    VBuyMax = Var(f'VBuyMax', lowBound=0)
    VSellMax = Var(f'VSellMax', lowBound=0)
    VMaxBuySell = Var(f'VMaxBuySell', lowBound=0)

    # Is buying/selling at each t
    VisBuy = {s: Var('VisBuy_{}'.format(s), cat='Binary') for s in time}
    VisSell = {s: Var('VisSell_{}'.format(s), cat='Binary') for s in time}
    VisBuySellMid = {s: Var('VisMid_{}'.format(s), cat='Binary') for s in time}
    # how much to VBuy/VSell at each t
    VBuy = {q: Var(f'VBuy_{q}', lowBound=0) for q in time}
    VSell = {q: Var(f'VSell_{q}', lowBound=0) for q in time}
    VBuyActual = {q: Var(f'VBuyActual_{q}', lowBound=0) for q in time}
    VSellActual = {q: Var(f'VSellActual_{q}', lowBound=0) for q in time}

    # SOC for vec at each t
    VSoc = {q: Var(f'VSoc_{q}', lowBound=0, upBound=39) for q in time}
    # Solar generation at each t
    VSolar = {q: Var(f'VSolar_{q}', lowBound=0) for q in time}

    # Is charging/discharging at each ev
    VisC = {s: Var('VCharge_{}'.format(s), cat='Binary') for s in ev}
    VisD = {s: Var('VDischarge_{}'.format(s), cat='Binary') for s in ev}
    VisCDMid = {s: Var('VDisCDMid_{}'.format(s), cat='Binary') for s in ev}
    # Hot much to charge/discharge at each t
    VptPlus = {q: Var(f'VptPlus_{q}', lowBound=0, upBound=3.7) for q in time}
    VptMinus = {q: Var(f'VptMinus_{q}', lowBound=0, upBound=3.7) for q in time}
    VptPlusActual = {q: Var(f'VptPlusActual_{q}', lowBound=0, upBound=3.7) for q in time}
    VptMinusActual = {q: Var(f'VptMinusActual_{q}', lowBound=0, upBound=3.7) for q in time}

    Vsmall4 = Var(f'Vsmall4', lowBound=4, upBound=8)
    Vsmall8 = Var(f'Vsmall8', lowBound=8)
    VbigNum = Var(f'VbigNum', lowBound=8)

    points = [0, 4, Vsmall4, 8, Vsmall8, VbigNum]

    Vx = Var(f'Vx', lowBound=0)
    Vy = Var(f'Vy', lowBound=0)
    Vw = {s: Var('Vw_{}'.format(s), lowBound=0, upBound=1) for s in points}
    Vu = {s: Var('Vu_{}'.format(s), cat='Binary') for s in points}


    # Objective function
    m += VBuyMax
    m += VSellMax
    m += VMaxBuySell
    m += lpSum((VBuyActual[t] * buyP[t]) for t in time) + Vy - lpSum((VSellActual[t] * sellP[t]) for t in time)

    # Constraints

    m += lpSum(Vu[e] for e in points) == 1
    m += lpSum(Vw[e] for e in points) == 1

    m += Vx == lpSum(points_n[t] * Vw[points[t]] for t in range(len(points)))
    m += Vy == lpSum(netfee[t] * Vw[points[t]] for t in range(len(points)))
    m += Vw[points[0]] <= Vu[points[0]]
    m += Vw[points[-1]] <= Vu[points[-2]]
    m += Vx == VMaxBuySell
    for i in range(len(points)):
        if points[i] != points[0] and points[i] != points[-1]:
            m += Vw[points[i]] <= Vu[points[i - 1]] + Vu[points[i]]

    m += VMaxBuySell >= VBuyMax
    m += VMaxBuySell >= VSellMax
    M = 50
    for t in time:
        m += VBuyMax >= VBuyActual[t]
        m += VSellMax >= VSellActual[t]

        # if VisBuy[t]/VisSell[t] is 0, then VBuyActual[t]/VSellActual[t] must be 0
        # VisBuy[t] and VisSell[t] can be 0 at same time, cannot be 1 at same time.
        m += VisBuy[t] <= 1 - VisBuySellMid[t]
        m += VisBuy[t] <= 1 - VisSell[t]
        m += VisBuy[t] >= 1 - VisBuySellMid[t] + 1 - VisSell[t] - 1

        m += VBuyActual[t] <= M * VisBuy[t]
        m += VBuyActual[t] <= VBuy[t]
        m += VBuy[t] - VBuyActual[t] <= M * (1 - VisBuy[t])

        m += VSellActual[t] <= M * VisSell[t]
        m += VSellActual[t] <= VSell[t]
        m += VSell[t] - VSellActual[t] <= M * (1 - VisSell[t])

        # ele into house == ele out house(+ consume)
        m += appsCost[t] + VptPlusActual[t] + VSellActual[t] == VSolar[t] + VptMinusActual[t] + VBuyActual[t]

        # ele into house fit appliances
        m += VSolar[t] + VptMinusActual[t] + VBuyActual[t] >= appsCost[t]

        # solar generation upBound
        m += VSolar[t] <= solarGen[t]

        if t not in ev:
            m += VptPlus[t] == 0
            m += VptMinus[t] == 0
            m += VptPlusActual[t] == 0
            m += VptMinusActual[t] == 0
            if t != 10 and t != 20:
                m += VSoc[t] == 0

    m += VSoc[ev[0]] == 30
    for t in ev:
        # if VisC[t]/VisD[t] is 0, then VptPlusActual[t]/VptMinusActual[t] must be 0
        # VisC[t] and VisD[t] can be 0 at same time, cannot be 1 at same time.
        m += VisC[t] <= 1 - VisCDMid[t]
        m += VisC[t] <= 1 - VisD[t]
        m += VisC[t] >= 1 - VisCDMid[t] + 1 - VisD[t] - 1

        m += VptPlusActual[t] <= M * VisC[t]
        m += VptPlusActual[t] <= VptPlus[t]
        m += VptPlus[t] - VptPlusActual[t] <= M * (1 - VisC[t])

        m += VptMinusActual[t] <= M * VisD[t]
        m += VptMinusActual[t] <= VptMinus[t]
        m += VptMinus[t] - VptMinusActual[t] <= M * (1 - VisD[t])

        m += VSoc[t + 1] == VSoc[t] + (0.9 * VptPlusActual[t]) - (VptMinusActual[t] * 10 / 9)

    m += VSoc[ev[-1] + 1] >= 33

    # Solving
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    Cost = 0
    vbuy = []
    vsell = []
    for t in time:
        vbuy.append(VBuyActual[t].value())
        vsell.append(VSellActual[t].value())
        Cost = Cost + (VisBuy[t].value() * VBuyActual[t].value() * buyP[t]) - (
                    VisSell[t].value() * VSellActual[t].value() * sellP[t])
    # Solution
    output["MILP status"].append(pulp.LpStatus[m.status])

    # Round the results to 1 decimal

    # Network fee (in $)
    output["network fee"] = round(Vy.value(), 1)  # <to be replaced by the student>
    # Electricity cost of the smart home (in $)
    output["electricity cost"] = round(Cost + Vy.value(), 1)  # <to be replaced by the student>
    # Electricity bought (in kWh)
    output["electricity bought"] = round(sum(vbuy), 1)  # <to be replaced by the student>
    # Electricity sold (in kWh)
    output["electricity sold"] = round(sum(vsell), 1)  # <to be replaced by the student>

    # <to be completed by the student>
    for t in range(len(data["solar"])):
        # Solar generation (in kW)
        output["solar"][t] = round(VSolar[t].value(), 1)  # <to be replaced by the student>
        # State-of-charge (in kWh)
        output["SOC"][t] = round(VSoc[t].value(), 1)  # <to be replaced by the student>
        # Battery charging (in kW)
        output["Charging"][t] = round(VptPlus[t].value(), 1)  # <to be replaced by the student>
        # Battery discharging (in kW)
        output["Discharging"][t] = round(VptMinus[t].value(), 1)  # <to be replaced by the student>

    return output


# Main ------------------------------------------------------------------------

if __name__ == "__main__":
    # Input data
    data = OrderedDict()

    # Electricity consumption of appliances (in kW) for 24 hours [0,....,23h]
    data["appiances"] = [3.3, 2.6, 3.4, 3.2, 2.3, 2.5, 2.7, 1.7, 2.5, 3.1, 1.8, 2.4, 1.7, 0.3, 0.8, 0.9, 0.2, 4.2, 2.1,
                         3.8, 4.0, 3.8, 3.2, 4.0]
    # Electricity generation of the rooftop solar system (in kW) for 24 hours [0,....,23h]
    data["solar"] = [0, 0, 0, 0, 0, 0, 0, 0, 0.4, 1.4, 3.1, 3.4, 2.0, 0.6, 0.4, 0.4, 0.3, 0.1, 0, 0, 0, 0, 0, 0]
    # Electricity buying price (in $/kWh) for 24 hours [0,....,23h]
    data["buying price"] = [0.1964, 0.1024, 0.1964, 0.1024, 0.1024, 0.1024, 0.1964, 0.1024, 0.1964, 0.2246, 0.1964,
                            0.1682, 0.1682,
                            0.1682, 0.1682, 0.1682, 0.1682, 0.1682, 0.2246, 0.2246, 0.1964, 0.1682, 0.1024, 0.1024]
    # Electricity selling price (in $/kWh) for 24 hours [0,....,23h]
    data["selling price"] = [0.053504, 0.053504, 0.0528, 0.050912, 0.04708, 0.044376, 0.0448, 0.048872, 0.048808,
                             0.0488, 0.0504,
                             0.0504, 0.051224, 0.052704, 0.052008, 0.048672, 0.048872, 0.050464, 0.053328, 0.053648,
                             0.053808,
                             0.054456, 0.053664, 0.051224]
    # Electric vehicle availability to charge and discharge
    data["EV time"] = [*range(10, 20, 1)]

    # Output data to be printed
    output = OrderedDict()

    # State-of-charge (in kWh)
    output["SOC"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Battery charging (in kW)
    output["Charging"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Battery discharging (in kW)
    output["Discharging"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Solar generation (in kW)
    output["solar"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Operating hours of the new appliance
    output["new appliance"] = []
    # Electricity cost of the house (in $)
    output["electricity cost"] = 0.0
    # Electricity newtork fee (in $)
    output["network fee"] = 0.0
    # Electricity bought (in kWh)
    output["electricity bought"] = 0.0
    # Electricity sold (in kWh)
    output["electricity sold"] = 0.0
    # Saves MILP status
    output["MILP status"] = []

    # Question 1
    # output = function_1(data, output)

    # Question 2
    # output = function_2(data, output)

    # Question 3
    output = function_3(data, output)

    # Print outputs
    print("MILP Status:", output["MILP status"][0])
    print("Battery state-of-charge(kWh):", output["SOC"])
    print("Battery charging (kW):", output["Charging"])
    print("Battery discharging (kW):", output["Discharging"])
    print("Solar generation (kW):", output["solar"])
    print("Electricity bought (kWh):", output["electricity bought"])
    print("Electricity sold (kWh):", output["electricity sold"])
    print("New appliance (h):", output["new appliance"])  # Only for question 2
    print("Electricity newtork fee ($):", output["network fee"])  # Only for question 3
    print("Electricity cost ($):", output["electricity cost"])
