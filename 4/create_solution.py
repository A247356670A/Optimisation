from pulp import PULP_CBC_CMD

from model import ModelBuilder
from nurse import RosteringProblem, save_roster, read_costs
from sys import argv

def create_solution(seed):
    '''
    Creates a feasible solution based on the specified seed.
    Different seeds should generally lead to different solutions.

    TODO: Implement this method.
    '''
    prob = RosteringProblem()
    costs = read_costs(prob)
    # print(costs)
    mb = ModelBuilder(prob)
    print(mb)
    model = mb.build_model(costs)
    # print(model)

    # if res != 1:
    #     print("No solution found")
    sol = mb.extract_solution()
    print(sol)
    return sol

if __name__ == '__main__':

    arg = 0 # default seed
    if len(argv) > 1:
        arg = int(argv[1])
    roster = create_solution(arg)
    print(roster)
    # Sanity check: making sure the roster is feasible
    prob = RosteringProblem()
    feasibility = prob.is_feasible(roster)
    if feasibility != None:
        print(f'roster is not feasible ({feasibility})')
        exit(0)

    save_roster(roster)

# eof
