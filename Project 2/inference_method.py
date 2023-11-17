from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE

def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    ### START: Your code
    inferred = dict()
    count = dict()

    if query not in symbols_list:
        return False
    
    for symbol in symbols_list:
        inferred[symbol] = False

    for clause in KB_clauses:
        count[clause.conclusion] = len(clause.body)

    while known_symbols != []:
        p = known_symbols.pop(0)

        if p == query:
            return True

        if inferred[p] == False:
            inferred[p] = True

            for clause in KB_clauses:
                if p in clause.body:
                    count[clause.conclusion] -= 1
                
                if count[clause.conclusion] == 0:
                    known_symbols.append(clause.conclusion)

    # works but slower
    # difference = 1

    # while difference != 0:
    #     l_initial = len(known_symbols)

    #     for clause in KB_clauses:
    #         if (clause.body and known_symbols) == known_symbols:
    #             known_symbols.append(clause.conclusion)

    #     if query in known_symbols:
    #         return True

    #     difference = l_initial - len(known_symbols)
   
    return False


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)
    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
