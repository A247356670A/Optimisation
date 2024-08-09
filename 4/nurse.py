from typing import List, Set, Dict
import re, os
from datetime import datetime

SHIFT_OFFDUTY = 'F'
SHIFT_MORNING = 'A'
SHIFT_AFTERNOON = 'P'
SHIFT_NIGHT = 'N'
SHIFTS = {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT, SHIFT_OFFDUTY}
DAYS_PER_WEEK = 7

def consecutive_numbers(list: List, length: int, elements: Set) -> Dict[int,int]:
    '''
      Returns a dictionary { s1: l1, s2: l2, etc.}
      where each si is an index of the list and li is a length
      such that [list] contains a sequence of li objects from [elements] starting at location [si].
      [length] indicates the length of relevant items in [list] 
      (that's in case you want the list to contain some sort of annotation; 
      just assume [length] == len(list)).
      This assumes that [list] is wrapped (the list restarts after the end of the list).
      Consequently it ignores the beginning of the list and the end of the list.

      For instance, if [list] is AAAXXAAAXA and [elements] is {A}
      then this method will return: {5->3, 9->4, 15->3}
      because there is a sequence of 3 As starting at position 5 (positions 5, 6, 7), 
      4 As starting at position 9 (positions 9, 0, 1, 2), 
      and 3 As starting at position 15 (position 5, 6, 7 similar to the first one when making a second pass).
      Importantly, the first sequence (length 3, starting at position 0)
      and the last sequence (length 1, starting at position 9) are ignored.
    '''
    result = {}

    # We iterate from position 0 to position len(list)*2 and count the number of occurrence.  We reset when we see something else
    have_seen_something_else = False
    nb_consecutives = 0 # Only start counting after we have seen something from outside [elements]

    for i in range(length*2):
        element = list[i%length]
        if element in elements:
            if have_seen_something_else:
                nb_consecutives += 1
        
        else: # element is not in elements
            have_seen_something_else = True
            if nb_consecutives != 0:
                start = i - nb_consecutives
                result[start] = nb_consecutives
            nb_consecutives = 0

    return result

class RosteringProblem:
    '''
    Definition of a rostering problem.  
    It is assumed that the following aspects are constant: 
    * A rostering solution includes 14 days.  The days [1,5] and [8,12] are week days, and the other days are week-end days.
    '''

    def __init__(self):
        self._nb_nurses = 16
        self._nb_weeks = 3
        self._min_nb_consecutive_days = 2
        self._max_nb_consecutive_days = 6
        self._nb_off_duty_days = 6
        self._min_nb_consecutive_work_days = 3
        self._max_nb_consecutive_work_days = 7
        self._min_nb_consecutive_off_days = 1
        self._max_nb_consecutive_off_days = 3
        self._shift_requirements = {SHIFT_MORNING:4, SHIFT_AFTERNOON:3, SHIFT_NIGHT:2}

    def get_nb_nurses(self):
        return self._nb_nurses
    def is_feasible(self, roster: List[str]) -> str:
        '''
            Indicates whether the specified roster is a feasible solution to the problem.
            A roster is defined as a vector of strings.
            roster[i] represents the roster of Nurse i (i in [0,...,NB_NURSES-1]).  
            roster[i][d] represents the shift of Nurse i on Day d (d in [0,...,NB_DAYS-1]), i.e., some element from {'F', 'A', 'P', 'N'}.
            Any other element is ignored (could be used to annotate a solution).

            This method returns a string describing one reason why the specified roster is not valid.  
            In other words, the roster is feasible iff this method returns None.
            In practice, you want probably want to create classes to represent these errors.
        '''
        # Constraint C0: contains the right number of rows and columns.
        if len(roster) < self._nb_nurses:
            return f'Not enough nurses ({len(roster)})'
        for i in range(self._nb_nurses):
            nurse = roster[i]
            if len(nurse) < self._nb_weeks * DAYS_PER_WEEK:
                return f'Wrong shift-length for nurse {i} ({len(nurse)})'

        # Constraint C1: shift is in {F,A,P,N}
        for i in range(self._nb_nurses):
            for d in range(self._nb_weeks * DAYS_PER_WEEK):
                shift = roster[i][d]
                if not shift in SHIFTS:
                    return f'Wrong type of shift for nurse {i} on day {d} ({shift})'

        # Constraint C2: 6 off-duty days per nurse
        for i in range(self._nb_nurses):
            nb_off_duty_days = len( [ d for d in range(self._nb_weeks * DAYS_PER_WEEK) if roster[i][d] == SHIFT_OFFDUTY ] )
            if nb_off_duty_days != self._nb_off_duty_days:
                return f'Wrong number of off-duty days for nurse {i} ({nb_off_duty_days})'

        # Constraint C3: number of consecutive days in a given shift
        for i in range(self._nb_nurses):
            for shift_type in {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT}:
                consecutives = consecutive_numbers(roster[i], self._nb_weeks * DAYS_PER_WEEK, shift_type)
                for k,v in consecutives.items():
                    if v < self._min_nb_consecutive_days or v > self._max_nb_consecutive_days:
                        return f'Wrong number of consecutive days for shift of nurse {i} starting from day {k}'

        # Constraint C4: Consecutive number of work days for a nurse
        for i in range(self._nb_nurses):
            consecutives = consecutive_numbers(roster[i], self._nb_weeks * DAYS_PER_WEEK, {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT})
            for k,v in consecutives.items():
                if v < self._min_nb_consecutive_work_days or v > self._max_nb_consecutive_work_days:
                    return f'Wrong number of consecutive work days for shift of nurse {i} starting from day {k}'

        # Constraint C5: Consecutive number of off-duty days for a nurse
        for i in range(self._nb_nurses):
            consecutives = consecutive_numbers(roster[i], self._nb_weeks * DAYS_PER_WEEK, {SHIFT_OFFDUTY})
            for k,v in consecutives.items():
                if v < self._min_nb_consecutive_off_days or v > self._max_nb_consecutive_off_days:
                    return f'Wrong number of consecutive off-duty days for shift of nurse {i} starting from day {k}'

        # Constraint C6: at least one week-end off-duty day
        weekenddays = (
            [   5+(DAYS_PER_WEEK*w) for w in range(self._nb_weeks) ] # Saturdays
            + [ 6+(DAYS_PER_WEEK*w) for w in range(self._nb_weeks) ] # Sundays
            )
        for i in range(self._nb_nurses):
            weekendshifts = { roster[i][d] for d in weekenddays }
            if not SHIFT_OFFDUTY in weekendshifts:
                return f'Nurse {i} does not have an off-duty day on week-ends'
        
        # Constraint C7: number of nurses in each shift
        for shift_type, min_nb in self._shift_requirements.items():
            for d in range(self._nb_weeks * DAYS_PER_WEEK):
                nurses = [ i for i in range(self._nb_nurses) if roster[i][d] == shift_type ]
                if len(nurses) < min_nb:
                    return f'Not enough nurses on shift {shift_type} for day {d}'

        # Constraint C8: order of shift (morning -> night -> afternoon)
        for i in range(self._nb_nurses):
            previous_work_shift = None
            for d in range(2 * self._nb_weeks * DAYS_PER_WEEK):
                shift = roster[i][d%(self._nb_weeks*DAYS_PER_WEEK)]
                if shift == SHIFT_OFFDUTY:
                    continue
                if shift == SHIFT_MORNING and previous_work_shift == SHIFT_NIGHT:
                    return f'Wrong shift order for nurse {i} on day {d} ({previous_work_shift} -> {shift})'
                if shift == SHIFT_NIGHT and previous_work_shift == SHIFT_AFTERNOON:
                    return f'Wrong shift order for nurse {i} on day {d} ({previous_work_shift} -> {shift})'
                if shift == SHIFT_AFTERNOON and previous_work_shift == SHIFT_MORNING:
                    return f'Wrong shift order for nurse {i} on day {d} ({previous_work_shift} -> {shift})'
                previous_work_shift = shift

        return None

    def cost(self, roster, costs):
        '''
          Returns the cost associated with the specified roster for the specified cost function.
        '''
        result = 0
        for i in range(self._nb_nurses):
            for d in range(self._nb_weeks * DAYS_PER_WEEK):
                shift = roster[i][d]
                cost_map = costs[i][d%DAYS_PER_WEEK]
                if shift in cost_map:
                    result += cost_map[shift]
        return result

    def print(self, roster):
        '''
          Prints the roster.  Each line is a nurse.
        '''
        for i in range(self._nb_nurses):
            print(roster[i][0:self._nb_weeks*DAYS_PER_WEEK])

def save_roster(roster, dir='.'):
    '''
    Saves the specified roster in the specified directory.  
    The filename is the time when the roster is saved.
    '''
    # Create a file based on the current time
    filename = os.path.join(dir,re.subn(':|\.| |-','_',str(datetime.now()))[0] + ".rost")
    with open(filename,'w') as f:
        for line in roster:
            f.write(line)
            f.write('\n')

def load_roster(filename):
    '''
    Loads a roster saved in the specified file.
    '''
    with open(filename) as f:
        return f.readlines()

def load_last_roster(dir='.'):
    '''
    Loads the last saved roster from the specified directory.  
    To determine which file to read, this method grabs all the files whose name starts with '2022' and ends with '.rost', 
    then sorts them alphabetically, and chooses the last one.  
    If you want to make copies of your existing files, 
    make sure that the name of these copies do not start with '2022' 
    to avoid unexpected results from this method.  
    '''
    filenames = os.listdir(dir)
    rosterfilenames = [ f for f in filenames if f.startswith('2022') and f.endswith('.rost')]
    rosterfilenames.sort(reverse=True)
    filename = os.path.join(dir, rosterfilenames[0]) # Will throw an error if the array is empty
    return load_roster(filename)

def read_costs(prob=RosteringProblem):
    '''
      Reads the cost for each nurse, day, and shift from the cost file.  
      This method first requires you to modify and run `create_costs.py` (you can run that file multiple times).
      The result is a table that indicates the costs for a nurse to work during a shift.  
      For instance, Nurse 5 being scheduled to work during day 16 in the Morning shift 
      induces the cost `read_costs()[5][16%7][SHIFT_MORNING].
      The cost is only defined for the work shift (in other words, the cost for SHIFT_OFFDUTY is 0).
    '''
    cost_list = []
    with open('costs.rcosts') as f:
        line = f.readline()
        cost_list = [ float(f) for f in re.findall('0\.\d*', line)]

    costs = []
    k = 0
    for _i in range(prob.get_nb_nurses()):
        nurse_costs = []
        for _d in range(DAYS_PER_WEEK):
            day_costs = {}
            for shift_type in [SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT]:
                day_costs[shift_type] = cost_list[k]
                k += 1
            nurse_costs.append(day_costs)
        costs.append(nurse_costs)
    return costs

# eof
