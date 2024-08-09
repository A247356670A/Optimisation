import random
from nurse import RosteringProblem, DAYS_PER_WEEK, SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT

if __name__ == '__main__':
    MY_ID = 1818576
    random.seed(MY_ID)

    with open('costs.rcosts', 'w') as f:
        prob = RosteringProblem()
        for i in range(prob._nb_nurses):
            for d in range(DAYS_PER_WEEK):
                for shift_type in [SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT]:
                    f.write(str(random.random())) # chooses a random number between 0 and 1
                    f.write(' ')

# eof
