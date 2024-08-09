class Neighbourhood:
    '''
    A neighbourhood is a class that computes explicitly neighbours of a roster.
    '''

    def neighbours(self, roster): 
        '''
          Returns a list of feasible rosters that are neighbours of this roster.
        '''
        pass

class SwapNeighbourhood(Neighbourhood):
    '''
      Example of a neighbourhood.
      A roster r' is a neighbour of a roster r 
      if it can be obtained by swapping the schedule of exactly two nurses.
    '''
    def __init__(self, prob) -> None:
        self._prob = prob

    def neighbours(self, roster):
        result = []
        for i in range(self._prob._nb_nurses):
            for j in range(i+1, self._prob._nb_nurses):
                # Swaps the schedules of nurses i and j.
                new_roster = [ line for line in roster] # notice that we can do that because strings are immutable (no risk of side effects)
                new_roster[i] = roster[j]
                new_roster[j] = roster[i]
                #if not self._prob.is_feasible(new_roster) == None:
                #    continue
                # No need to test for feasibility: this new roster is guaranteed to be feasible (assuming the specified one was feasible).
                result.append(new_roster)
        return result

# TODO: Define more neighbourhoods, and remove this comment.

# eof
