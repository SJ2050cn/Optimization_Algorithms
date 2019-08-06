# ----------------------------------------------------------------------
# cuckoo_search via levy flight for global optimization
#                                                          -- coded by SJ2050 in 2019
#-----------------------------------------------------------------------
import numpy as np
import scipy.special as sc_special

version = '1.0.0'

def cuckoo_search(n, m, fit_func, lower_boundary, upper_boundary, iter_num = 100,pa = 0.25, beta = 1.5, step_size = 0.1):
    """
    Cuckoo search function
    ---------------------------------------------------
    Input parameters:
        n: Number of nests
        m: Number of dimensions
        fit_func: User defined fitness evaluative function
        lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
        upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
        iter_num: Number of iterations (default: 100) 
        pa: Possibility that hosts find cuckoos' eggs (default: 0.25)
        beta: Power law index (note: 1 < beta < 2) (default: 1.5)
        step_size:  Step size scaling factor related to the problem's scale (default: 0.1)
    Output:
        The best solution and its value
    """
    # get initial nests' locations 
    nests = generate_nests(n, m, lower_boundary, upper_boundary)
    fitness = calc_fitness(fit_func, nests)

    # get the best nest and record it
    best_nest_index = np.argmax(fitness)
    best_fitness = fitness[best_nest_index]
    best_nest = nests[best_nest_index].copy()

    for _ in range(iter_num):
        nests = update_nests(fit_func, lower_boundary, upper_boundary, nests, best_nest, fitness, step_size)
        nests = abandon_nests(nests, lower_boundary, upper_boundary, pa)
        fitness = calc_fitness(fit_func, nests)
        
        max_nest_index = np.argmax(fitness)
        max_fitness = fitness[max_nest_index]
        max_nest = nests[max_nest_index]

        if (max_fitness > best_fitness):
            best_nest = max_nest.copy()
            best_fitness = max_fitness

    return (best_nest, best_fitness)

def generate_nests(n, m, lower_boundary, upper_boundary):
    """
    Generate the nests' locations
    ---------------------------------------------------
    Input parameters:
        n: Number of nests
        m: Number of dimensions
        lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
        upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
    Output:
        generated nests' locations
    """
    lower_boundary = np.array(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    nests = np.empty((n, m))

    for each_nest in range(n):
        nests[each_nest] = lower_boundary + np.array([np.random.rand() for _ in range(m)]) * (upper_boundary - lower_boundary)

    return nests

def update_nests(fit_func, lower_boundary, upper_boundary, nests, best_nest, fitness, step_coefficient):
    """
    This function is to get new nests' locations and use new better one to replace the old nest
    ---------------------------------------------------
    Input parameters:
        fit_func: User defined fitness evaluative function
        lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
        upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
        nests: Old nests' locations 
        best_nest: Nest with best fitness
        fitness: Every nest's fitness
        step_coefficient:  Step size scaling factor related to the problem's scale (default: 0.1)
    Output:
        Updated nests' locations
    """
    lower_boundary = np.array(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    n, m = nests.shape
    # generate steps using levy flight
    steps = levy_flight(n, m, 1.5)
    new_nests = nests.copy()

    for each_nest in range(n):
        # coefficient 0.01 is to avoid levy flights becoming too aggresive
        # and (nest[each_nest] - best_nest) could let the best nest be remained
        step_size = step_coefficient * steps[each_nest] * (nests[each_nest] - best_nest)
        step_direction = np.random.rand(m)
        new_nests[each_nest] += step_size * step_direction
        # apply boundary condtions
        new_nests[each_nest][new_nests[each_nest] < lower_boundary] = lower_boundary[new_nests[each_nest] < lower_boundary]
        new_nests[each_nest][new_nests[each_nest] > upper_boundary] = upper_boundary[new_nests[each_nest] > upper_boundary]

    new_fitness = calc_fitness(fit_func, new_nests)
    nests[new_fitness > fitness] = new_nests[new_fitness > fitness]
    
    return nests

def abandon_nests(nests, lower_boundary, upper_boundary, pa):
    """
    Some cuckoos' eggs are found by hosts, and are abandoned.So cuckoos need to find new nests.
    ---------------------------------------------------
    Input parameters:
        nests: Current nests' locations
        lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
        upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
        pa: Possibility that hosts find cuckoos' eggs
    Output:
        Updated nests' locations
    """
    lower_boundary = np.array(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    n, m = nests.shape
    for each_nest in range(n):
        if (np.random.rand() < pa):
            step_size = np.random.rand() * (nests[np.random.randint(0, n)] - nests[np.random.randint(0, n)])
            nests[each_nest] += step_size
            # apply boundary condtions
            nests[each_nest][nests[each_nest] < lower_boundary] = lower_boundary[nests[each_nest] < lower_boundary]
            nests[each_nest][nests[each_nest] > upper_boundary] = upper_boundary[nests[each_nest] > upper_boundary]
    
    return nests

def levy_flight(n, m, beta):
    """
    This function implements Levy's flight.
    ---------------------------------------------------
    Input parameters:
        n: Number of steps 
        m: Number of dimensions
        beta: Power law index (note: 1 < beta < 2)
    Output:
        'n' levy steps in 'm' dimension
    """
    sigma_u = (sc_special.gamma(1+beta)*np.sin(np.pi*beta/2)/(sc_special.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    sigma_v = 1

    u =  np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, sigma_v, (n, m))

    steps = u/((np.abs(v))**(1/beta))

    return steps

def calc_fitness(fit_func, nests):
    """
    calculate each nest's fitness
    ---------------------------------------------------
    Input parameters:
        fit_func: User defined fitness evaluative function
        nests:  Nests' locations
    Output:
        Every nest's fitness
    """
    n, m = nests.shape
    fitness = np.empty(n)

    for each_nest in range(n):
        fitness[each_nest] = fit_func(nests[each_nest])

    return fitness

if __name__=='__main__':
    def fit_func(nest):
        x, y = nest        
        return 3*(1-x)**2*np.e**(-x**2-(y+1)**2) - 10*(x/5-x**3-y**5)*np.e**(-x**2-y**2) - (np.e**(-(x+1)**2-y**2))/3

    best_nest, best_fitness = cuckoo_search(25, 2, fit_func, [-3, -3], [3, 3], step_size = 0.4)

    print('最大值为:%.5f, 在(%.5f, %.5f)处取到!'%(best_fitness, best_nest[0], best_nest[1]))
