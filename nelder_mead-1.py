import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

def rosenbrock2D(x,y):
    return (1 - x)**2.0 + 100*(y - x**2.0)**2.0

def rosenbrockNDforgraph(xvec):
    dim = xvec.shape[1]
    result = np.zeros(xvec.shape[0])
    for i in range(dim-1):
        result += rosenbrock2D(xvec[:,i], xvec[:,i+1])
    #print(result)
    return result

def rosenbrockND(xvec):
    dim = xvec.shape[1]
    result = np.zeros(xvec.shape[0])
    for i in range(dim-1):
        result += rosenbrock2D(xvec[:,i], xvec[:,i+1])
    #print(result)
    return result[0]

def bukin2D(xarr):
    x = xarr[:, 0]
    y = xarr[:, 1]
    return 100*np.sqrt(np.abs(y - 0.01*x**2.0)) + 0.01*np.abs(x + 10)

def camel3_2D(xarr):
    x1 = xarr[:, 0]
    x2 = xarr[:, 1]
    return 2*x1**2 -1.05*x1**4 + (1/6)*x1**6 + x1*x2 + x2**2

def camel6_2D(xarr):
    x1 = xarr[:, 0]
    x2 = xarr[:, 1]
    return (4 - 2.1*x1**2 + (1/3)*x1**4) * (x1**2) + x1*x2 + (-4 + 4*x2**2)*x2**2

phi = (-1 + 5**.5)/2

def gss(f, x0, x3,eps=1e-6):
    x1 = (1 - phi) * (x3 - x0) + x0
    x2 = (phi * (x3 - x0)) + x0

    f0, f1, f2, f3 = [f(x) for x in [x0, x1, x2, x3]]

    while (abs(x3 - x0)>2*eps):
        if f0 < f1 and f0 < f2 and f0 < f3:
            x3 = x2
            f3 = f2
            x2 = x1
            f2 = f1
            x1 = (1 - phi) * (x3 - x0) + x0
            f1 = f(x1)
        elif f3 < f2 and f3 < f1 and f3 < f0:
            x0 = x1
            f0 = f1
            x1 = x2
            f1 = f2
            x2 = (phi * (x3 - x0)) + x0
            f2 = f(x2)
        if f1 < f2:
            x3 = x2
            f3 = f2
            x2 = x1
            f2 = f1
            x1 = (1 - phi) * (x3 - x0) + x0
            f1 = f(x1)
        elif f1 >= f2:
            x0 = x1
            f0 = f1
            x1 = x2
            f1 = f2
            x2 = (phi * (x3 - x0)) + x0
            f2 = f(x2)
    min_idx = sorted([0,1,2,3],key=lambda x: [f0,f1,f2,f3][x])[0]
    min_x = [x0,x1,x2,x3][min_idx]
    min_f = [f0,f1,f2,f3][min_idx]
    return {"x":min_x,"fun":min_f}


def nelder_mead(f, x0, new_reflect = False, new_expand = False, new_contract = False, eps=1e-12, alpha=2.5, gamma=3.0, rho=.5, sigma=.5,):
    dim = x0.shape[1]
    corners = [x0]
    for i in range(dim):
        corners.append(x0+np.eye(dim)[i])
    # corners = [x0, x0 + np.array([[0,1]]), x0 + np.array([[1,0]])]
    simplex = [(x, f(x)) for x in corners]
    tris = []
    while (len(tris) <= 1 
        or np.linalg.norm((np.array(tris[-1]) - np.array([item[0] for item in simplex]))) > eps):
        #add the edges of the simplex to tris, to keep track of triangles
        tris.append([item[0] for item in simplex])

        # one iteration of Nelder-Mead
        # Step 1: sort the points
        simplex = sorted(simplex, key=lambda x: x[1])
        
        x1,f1 = simplex[0]
        x2last,f2last = simplex[-2]
        xlast,flast = simplex[-1]

        # step 2: find the center of the simplex
        if new_reflect == True:
            denom = sum([math.e**item[1] for item in simplex])
            print("denom:",denom)
            numer = [math.e**item[1] for item in simplex]
            print("numer:",numer)
            score = numer / denom
            index = np.arange(0,len(simplex),1)
            x0 = sum([score[i]*simplex[i][0] for i in index])
            print("simplex:",simplex)
            print("x0:",x0)
        else:
            x0 = sum([item[0] for item in simplex])/len(simplex)
        # print("x0:",x0)
        f0 = f(x0)

        # step 3: reflect worst point over the plane spanned by the other points
        xr = x0 + alpha*(x0 - xlast)
        fr = f(xr)
        
        if fr < f2last and fr >= f1:
            simplex[-1] = (xr,fr)
            print("reflect!")
            continue
        # step 4: expansion
        # test if the reflected point is the best so far
        if fr < f1:
            # if it is, push further in that direction
            if new_expand == True:
                direction = x0 - simplex[-1][0]
                myfunc = lambda var: f(xr + var * direction)
                result = gss(myfunc, 6 * np.linalg.norm(x0 + xr - x0), np.linalg.norm(x0))
                xe = xr + result['x'] * direction
            else:
                xe = x0 + gamma*(xr - x0)
            fe = f(xe)
            # if fe is better than fr
            if fe < fr:
                simplex[-1] = (xe,fe)
                print("expand fe < fr!")
            else:
                simplex[-1] = (xr,fr)
                print("expand fe >= fr!")
            continue

        # step 5: contraction
        if fr < flast:
            # compute outside contraction point
            if new_contract == True:
                direction = xr - x0
                myfunc = lambda var: f(x0 + var * direction)
                result = gss(myfunc, 0, 0.5)
                xc = x0 + result['x'] * direction
            else:
                xc = x0 + rho * (xr - x0)
            fc = f(xc)
            # compare contracted point to reflected point
            if fc < fr: 
                simplex[-1] = (xc,fc)
                print("contract 1!")
                continue
            else:
                pass
        else:
            # this means fr was worse than f3, so we try an interior contraction
            if new_contract == True:
                direction = xlast - x0
                myfunc = lambda var: f(x0 + var * direction)
                result = gss(myfunc, 0, 0.5)
                xc = x0 + result['x'] * direction
            else:
                xc = x0 + rho * (xlast - x0)
            fc = f(xc)
            if fc < flast:
                simplex[-1] = (xc,fc)
                print("contract 2!")
                continue
            else:
                pass

        # step 6: move all points other than x1, slightly closer to x1
        print("last resort!")
        simplex_new = [(x1,f1)]
        for i in range(1,len(simplex)):
            xnew = x1 + sigma*(simplex[i][0] - x1)
            fnew = f(xnew)
            simplex_new.append((xnew,fnew))
        simplex = simplex_new

    return tris,x0

def plot_func(axis, func, xmin, xmax, ymin, ymax):
    delta = 0.025
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    X2 = X.reshape(-1,1)
    Y2 = Y.reshape(-1,1)
    Z2 = func(np.concatenate([X2, Y2], 1))
    Z = Z2.reshape(X.shape)

    ax.contour(X, Y, Z, np.exp(np.linspace(-2,8,32)),colors='red')

def plot_tris(axis, triangle_list):
    cmap = matplotlib.colormaps["plasma"]
    for i,tri in enumerate(triangle_list):
        #print(np.stack(tri).shape)
        t2 = plt.Polygon(np.concatenate(tri),
                        color=cmap(i/len(triangle_list)),
                        alpha=.25,
                        zorder=5
        )
        axis.add_patch(t2)

def plot_ls_vs_standard_nelder_mead():
    exponents = np.arange(-12, -1, 0.2)
    eps_values = [10 ** e for e in exponents]
    func = rosenbrockND

    startPoint = np.random.random((1, 2))
    # comment out the lines below for randomness
    startPoint[0][0] = 0.48535871
    startPoint[0][1] = 0.20660502


    fig, ax = plt.subplots()
    ax.set(
        xscale = 'log',
        xlabel='epsilon value',
        ylabel='num of simplex moves made',
        title='Nelder Mead Performance for Epsilon Values: Rosenbrock function'
    )
    ax.invert_xaxis()

    ex_nelder_mead_moves = []
    co_nelder_mead_moves = []
    std_nelder_mead_moves = []

    for eps in eps_values:
        ex_tris,min1 = nelder_mead(func, startPoint, new_expand = True, eps = eps)
        co_tris,min1 = nelder_mead(func, startPoint, new_contract = True, eps = eps)
        std_tris,min2 = nelder_mead(func, startPoint, eps = eps)

        ex_nelder_mead_moves.append(len(ex_tris))
        co_nelder_mead_moves.append(len(co_tris))
        std_nelder_mead_moves.append(len(std_tris))

    ax.plot(eps_values, std_nelder_mead_moves, color='blue', label='Standard Nelder-Mead')
    ax.plot(eps_values, ex_nelder_mead_moves, color='red', label='Nelder-Mead w/ expansion line search')
    ax.plot(eps_values, co_nelder_mead_moves, color='green', label='Nelder-Mead w/ contraction line search')

    plt.legend()
    plt.show()

func = rosenbrockND
funcforgraph = rosenbrockNDforgraph
test = 2*np.random.random((1,2))
print("test:",test)
print("test.shape[1]:",test.shape[1])

tris,min = nelder_mead(func, test,True)


print("minimum:",min)

fig, ax = plt.subplots()
# plot_func(ax, funcforgraph, -3,3,-5,5)
plt.plot([1],[1],marker="o", markersize=5, markeredgecolor="red", markerfacecolor="blue")
plot_tris(ax, tris)
plt.show()
plot_ls_vs_standard_nelder_mead()