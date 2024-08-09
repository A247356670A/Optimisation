from pulp import LpProblem, LpVariable, lpSum

from nurse import DAYS_PER_WEEK, SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT, SHIFT_OFFDUTY

class ModelBuilder:
    '''
    Object used to create a pulp model for LNS.
    '''

    def __init__(self, prob) -> None:
        self._prob = prob
        self._nurses = range(0, prob._nb_nurses)
        self._nb_days = prob._nb_weeks*DAYS_PER_WEEK
        self._days = range(0,self._nb_days)
        self._shifts = {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT, SHIFT_OFFDUTY}
        self._workshifts = SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT

        saturdays = {5+w*DAYS_PER_WEEK for w in range(prob._nb_weeks)}
        sundays = {6+w*DAYS_PER_WEEK for w in range(prob._nb_weeks)}
        self._weekends = saturdays.union(sundays)

        self._choices = LpVariable.dicts("Choice", (self._nurses, self._days, self._shifts), cat="Binary")

    def min_number_of_consecutive_days_in_shifts(self, i, shifts, d, m, model):
        '''
          Adds to [model] the constraints that guarantee that
          if [d] is the start of a sequence of days in which Nurse [i] performs shifts from [shifts], 
          then this nurse performs a shift from [shifts] from day [d] until day [d+m-1]
        '''
        start = d + self._nb_days # Will avoid issues when talking about d-1

        # The constraint can be written as follows: 
        # # for all x in [1,...,m-1], 
        # # # if choice[i][d][s] = 1 for some s in shifts and choice[i][d-1][s] = 0 for all s in shifts, 
        # # # then choice[i][d+x][d] = 1 for some s in shifts
        # which can be rewriten
        # # for all x in [1,...,m-1], 
        # # # Sum_{s in shift} ( choice[i][d-1][s] + choice[i][d+x][s] - choice[i][d][s] ) >= 0
        # (i.e., choice[i][d][s] should not be the only one evaluating to 1).
        for x in range(1,m):
                model += ( 
                      lpSum([self._choices[i][(start-1)%self._nb_days][shift] for shift in shifts])
                    + lpSum([self._choices[i][(start+x)%self._nb_days][shift] for shift in shifts])
                    - lpSum([self._choices[i][(start)%self._nb_days][shift] for shift in shifts])
                    >= 0
                )

    def max_number_of_consecutive_days_in_shifts(self, i, shifts, d, m, model):
        '''
          Adds to [model] the constraint that guarantees 
          that there isn't a consecutive sequence of [m]+1 days starting in [m] 
          in which Nurse [i] always performs shifts from the specified set.
        '''
        start = d + self._nb_days # Will avoid issues when talking about day - xxx

        # The constraint can be written as follows: 
        # Sum_{dd in [d,d+m], s in shifts} choice[i][dd][s] <= m (in other words, not m+1).
        model += (
            lpSum([self._choices[i][(d+j)%self._nb_days][s] for j in range(m+1) for s in shifts])
            <= m
        )

    def build_model(self, costs):
        '''
          Returns a pulp model that contains the constraints for a solution
        '''
        model = LpProblem()

        # c1
        for i in self._nurses:
            for d in self._days:
                model += lpSum([self._choices[i][d][s] for s in self._shifts]) == 1
    
        # c2
        for i in self._nurses:
            model += lpSum([self._choices[i][d][SHIFT_OFFDUTY] for d in self._days]) == 6
    
        # c3
        for i in self._nurses:
            for shift_type in {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT}:
                for day in self._days:
                    self.min_number_of_consecutive_days_in_shifts(i, {shift_type}, day, self._prob._min_nb_consecutive_days, model)
                    self.max_number_of_consecutive_days_in_shifts(i, {shift_type}, day, self._prob._max_nb_consecutive_days, model)

        # c4
        for i in self._nurses:
            work_shifts = {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT}
            for day in self._days:
                self.min_number_of_consecutive_days_in_shifts(i, work_shifts, day, self._prob._min_nb_consecutive_work_days, model)
                self.max_number_of_consecutive_days_in_shifts(i, work_shifts, day, self._prob._max_nb_consecutive_work_days, model)

        # c5
        for i in self._nurses:
            for day in self._days:
                self.min_number_of_consecutive_days_in_shifts(i, {SHIFT_OFFDUTY}, day, self._prob._min_nb_consecutive_off_days, model)
                self.max_number_of_consecutive_days_in_shifts(i, {SHIFT_OFFDUTY}, day, self._prob._max_nb_consecutive_off_days, model)
    
        # c6
        for i in self._nurses:
            model += (
                lpSum( [self._choices[i][d][SHIFT_OFFDUTY] for d in self._weekends] )
                >= 1
            )

        # c7
        for shift_type, min_nb in self._prob._shift_requirements.items():
            for d in range(self._prob._nb_weeks * DAYS_PER_WEEK):
                model += (
                    lpSum( [self._choices[i][d][shift_type] for i in self._nurses ] )
                    >= min_nb
                )

        # c8
        # Cannot have shift1@t, shift2@(t+1), ..., shift2@(t+k), shift3@(t+k+1) in this order.
        # This is modelled by saying that the sum of these things is at most k+1 (out of k+2 elements)
        # Notice that k is limited by the max length of shift2 (here SHIFT_OFF)
        for i in self._nurses:
            for d in self._days:
                for (shift1,shift2,shift3) in [
                    (SHIFT_MORNING, SHIFT_OFFDUTY, SHIFT_AFTERNOON), 
                    (SHIFT_AFTERNOON, SHIFT_OFFDUTY, SHIFT_NIGHT), 
                    (SHIFT_NIGHT, SHIFT_OFFDUTY, SHIFT_MORNING), 
                ]:
                    for k in range(self._prob._max_nb_consecutive_off_days+1):
                        model += (
                            self._choices[i][d][shift1]
                            + self._choices[i][(d+k+1)%self._nb_days][shift3]
                            + lpSum( [self._choices[i][(d+j)%self._nb_days][shift2] for j in range(1,k+1) ] )
                            <= k+1
                        )
        
        # Optimisation function
        model += lpSum( 
            [costs[i][d%DAYS_PER_WEEK][shift]*self._choices[i][d][shift] 
            for i in self._nurses for d in self._days for shift in self._workshifts]
            )

        return model
    
    def extract_solution(self):
        '''
          Returns the solution computed for the model.
        '''
        result = []
        for i in self._nurses:
            nurse_roster = ''
            for d in self._days:
                for shift in self._shifts:
                    if self._choices[i][d][shift].value() == 1:
                        nurse_roster += shift
            result.append(nurse_roster)
        return result

# eof
