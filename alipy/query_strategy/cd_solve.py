import numpy as np

def obj_fcn(P, q, x):
    return np.sum( (0.5*np.dot(x, P) + q) * x )

def _solve_one(P, q, x, Px, i, j):
    x_i, x_j = x[i], x[j]
    if x_i == 0.0 and x_j == 0.0:
        return 0.0

    q_i, q_j = q[i], q[j]
    P_ii, P_ij, P_ji, P_jj = P[i,i], P[i,j], P[j,i], P[j,j]
    a = (P_ii - P_ij - P_ji + P_jj) # 2a
    if a <= 1e-20:
        return 0.0
    b = (Px[i] - Px[j] )  + q_i - q_j
    
    min_pt = -b/a
    if min_pt > 0.0:
        if min_pt < x_j:
            d = min_pt
        else:
            d = x_j
    elif min_pt < 0.0:
        if min_pt > -x_i:
            d = min_pt
        else:
            d = -x_i
    else:
        d = 0

    return d

def cd_solve_sub(P, q, x, idx_pool, pos_pool):
    n = len(idx_pool)
    if n <= 1:
        return x
    Px = np.dot(P, x)
    for _ in range(1):
        np.random.shuffle(idx_pool)
        #for idx in range(int(n/2)):
        for idx in range(n):
            #i, j = idx_pool[2*idx], idx_pool[2*idx+1]
            i = idx_pool[idx]
            j = np.random.choice(pos_pool)
            if i == j:
                continue
            d = _solve_one(P, q, x, Px, i, j)
            x[i], x[j] = x[i] + d, x[j] - d
            Px = Px + (P[i]-P[j])*d
    return x

def cd_solve(P, q):
    n = P.shape[0]
    x = np.ones(n)/n
    idx_pool = np.array(range(n))
    pos_pool = np.where(x>0)[0]

    # old_obj = obj_fcn(P, q, x)
    # print("init_obj = ", old_obj)
    
    x_tmp = cd_solve_sub(P, q, np.copy(x), idx_pool, pos_pool)
    while np.sum((x_tmp - x)**2) > 1e-12:
        x = x_tmp
        x_tmp = cd_solve_sub(P, q, np.copy(x), idx_pool, pos_pool)
        pos_pool = np.where(x_tmp>0)[0]
        # obj = obj_fcn(P, q, x)
        # print("obj = ", obj)
        #break
        
    return x

if __name__ == "__main__":
    N = 100
    np.random.seed(0)
    P = np.random.normal(size=[N,N])
    P = np.dot(P, P.T)
    q = np.random.normal(size=[N])
    x_ans = cd_solve(P, q)
    print("final obj: ", obj_fcn(P, q, x_ans))
    print("non zeros: ", len(np.where(x_ans > 0)[0]))
