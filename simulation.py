import numpy as np
import scipy

def max_call_payoff(K):
    # It is assumed that x.shape == n_paths * n_stocks, representing
    # the price of stocks at a specific time.
    return lambda x: np.clip(np.max(x, axis=-1)-K, 0, None)

def leaky_relu(a):
    return lambda x: np.clip(x, 0, None) - np.clip(-a * x, 0, None)

def generate_paths(init_price, n_stocks, n_dates=10, T=1,
                   n_paths=2000, rate=0.0, volatility=0.2):
    """Return an array (n_paths * n_stocks * n_dates) of prices
    according to the Black-Scholes model.

    """
    dt = T/n_dates

    #             ⎛⎛     2⎞         ⎞
    #             ⎜⎜    σ ⎟         ⎟
    # X  = x  Exp ⎜⎜μ - ──⎟ t + σ W ⎟
    #  t    0     ⎝⎝    2 ⎠        t⎠
    x = np.exp(
        (rate - (volatility**2)/2) * dt \
        + volatility * np.sqrt(dt) \
        * np.random.randn(n_paths, n_stocks, n_dates)
    )
    x = init_price * x.cumprod(-1)
    return x

def rlsm(init_price, n_stocks, discount=1,
         payoff=max_call_payoff(100), activation_fn=leaky_relu(0.5),
         hidden_nodes=20, n_dates=10, T=1, n_paths=2000):

    x = generate_paths(init_price, n_stocks, T=T,
                       n_dates=n_dates, n_paths=2*n_paths)

    p = np.empty((2*n_paths, n_dates))
    p[:, n_dates-1] = payoff(x[:,:,n_dates-1])

    A = np.random.randn(hidden_nodes, n_stocks)
    b = np.random.randn(hidden_nodes)

    theta = np.zeros((n_dates, hidden_nodes + 1))
    for t in range(n_dates-1, 0, -1):
        phi = np.hstack((
            activation_fn(x[:,:,t-1] @ A.T + b),
            np.ones(2*n_paths).reshape(-1,1)
        ))
        theta[t-1] = scipy.linalg.lstsq(
            phi[0:n_paths],
            discount * p[0:n_paths, t]
        )[0]
        idx = payoff(x[:,:,t-1]) >= phi @ theta[t-1]
        p[idx, t-1] = payoff(x[idx,:,t-1])
        p[np.logical_not(idx), t-1] = \
            discount * p[np.logical_not(idx), t]

    return max(payoff(init_price),
               discount * np.mean(p[n_paths:2*n_paths, 0]))


def rfqi(init_price, n_stocks, discount=1,
         payoff=max_call_payoff(100), activation_fn=leaky_relu(0.5),
         hidden_nodes=20, n_dates=10, T=1, n_paths=2000,
         n_epochs=100, tol=1e-3, verbose=False):

    # x.shape == (2*n_paths, n_stocks, n_dates)
    x = generate_paths(init_price, n_stocks, T=T,
                       n_dates=n_dates, n_paths=2*n_paths)

    # transform(x).shape == (2*paths, n_dates-1, n_stocks+2)
    def transform(x):
        m, d, N = x.shape
        y1 = np.transpose(x[:, :, 0:N-1], (0,2,1))
        y2 =  np.empty((m,N-1,2))
        for i in range(N-1):
            y2[:, i, 0] = i+1
            y2[:, i, 1] = N-i-1
        return np.concatenate((y1, y2), axis=2)

    xt = transform(x)

    p = np.empty((2*n_paths, n_dates))
    p[:, n_dates-1] = payoff(x[:,:,n_dates-1])

    A = np.random.randn(hidden_nodes, n_stocks + 2)
    b = np.random.randn(hidden_nodes)

    # phi.shape == (2*n_paths, n_dates-1, hidden_nodes+1)
    phi = np.concatenate(
        (activation_fn(xt @ A.T + b),
         np.ones((2*n_paths, n_dates-1, 1))),
        axis=2
    )
    theta = np.zeros(hidden_nodes + 1)
    for epoch in range(n_epochs):
        for t in range(1, n_dates):
            p[:, t-1] = np.maximum(
                payoff(x[:,:,t-1]),
                phi[:,t-1,:] @ theta
            )

        new_theta = scipy.linalg.lstsq(
            phi[0:n_paths, :, :].reshape(n_paths*(n_dates-1),
                                         hidden_nodes+1),
            discount * p[0:n_paths, 1:n_dates].ravel()
        )[0]

        delta_theta = scipy.linalg.norm(new_theta - theta)
        theta = new_theta

        if verbose:
            print("Epoch", 1+epoch, ":", "Δθ =", delta_theta)

        if delta_theta < tol:
            break

    prices = np.empty(n_paths)
    for i in range(n_paths):
        for t in range(n_dates):
            g = payoff(x[n_paths + i, :, t])
            if t == n_dates - 1:
                prices[i] = g * (discount ** t)
                break
            c = np.dot(phi[n_paths + i, t, :], theta)
            if g >= c:
                prices[i] = g * (discount ** t)
                break

    return max(payoff(init_price), discount * np.mean(prices))



inits = [80,100,120]
ds = [5,10,50,100,500,1000]

for d in ds:
    for x0 in inits:
        ps = [rlsm(x0, d, n_paths=10000) for _ in range(10)]
        mean = np.round(np.mean(ps), decimals=2)
        std = np.round(np.std(ps), decimals=2)
        print("rlsm(", x0, ",", d,") =", mean, "(", std, ")")

# Output:
# rlsm( 80  , 5    ) = 4.64   ( 0.13 )
# rlsm( 100 , 5    ) = 23.94  ( 0.2  )
# rlsm( 120 , 5    ) = 48.32  ( 0.21 )
# rlsm( 80  , 10   ) = 8.24   ( 0.08 )
# rlsm( 100 , 10   ) = 32.81  ( 0.16 )
# rlsm( 120 , 10   ) = 59.3   ( 0.13 )
# rlsm( 80  , 50   ) = 21.81  ( 0.06 )
# rlsm( 100 , 50   ) = 52.26  ( 0.09 )
# rlsm( 120 , 50   ) = 82.76  ( 0.11 )
# rlsm( 80  , 100  ) = 28.4   ( 0.08 )
# rlsm( 100 , 100  ) = 60.47  ( 0.08 )
# rlsm( 120 , 100  ) = 92.57  ( 0.15 )
# rlsm( 80  , 500  ) = 43.07  ( 0.06 )
# rlsm( 100 , 500  ) = 78.81  ( 0.1  )
# rlsm( 120 , 500  ) = 114.55 ( 0.08 )
# rlsm( 80  , 1000 ) = 49.12  ( 0.09 )
# rlsm( 100 , 1000 ) = 86.37  ( 0.06 )
# rlsm( 120 , 1000 ) = 123.66 ( 0.12 )

inits = [80,100,120]
ds = [5,10,50,100,500,1000]

for d in ds:
    for x0 in inits:
        ps = [rfqi(x0, d, n_paths=10000,
                   hidden_nodes=min(d, 20), n_epochs=200,
                   verbose=True) for _ in range(10)]
        mean = np.round(np.mean(ps), decimals=2)
        std = np.round(np.std(ps), decimals=2)
        print("rfqi(", x0, ",", d,") =", mean, "(", std, ")")

# Output:
# rfqi( 80  , 5    ) = 4.98   ( 0.06 )
# rfqi( 100 , 5    ) = 24.53  ( 0.09 )
# rfqi( 120 , 5    ) = 49.25  ( 0.33 )
# rfqi( 80  , 10   ) = 8.81   ( 0.08 )
# rfqi( 100 , 10   ) = 33.61  ( 0.14 )
# rfqi( 120 , 10   ) = 60.36  ( 0.17 )
# rfqi( 80  , 50   ) = 22.84  ( 0.07 )
# rfqi( 100 , 50   ) = 53.51  ( 0.13 )
# rfqi( 120 , 50   ) = 84.12  ( 0.17 )
# rfqi( 80  , 100  ) = 29.26  ( 0.05 )
# rfqi( 100 , 100  ) = 61.6   ( 0.08 )
# rfqi( 120 , 100  ) = 93.81  ( 0.19 )
# rfqi( 80  , 500  ) = 43.64  ( 0.07 )
# rfqi( 100 , 500  ) = 79.55  ( 0.12 )
# rfqi( 120 , 500  ) = 115.42 ( 0.15 )
# rfqi( 80  , 1000 ) = 49.63  ( 0.05 )
# rfqi( 100 , 1000 ) = 87.05  ( 0.11 )
# rfqi( 120 , 1000 ) = 124.42 ( 0.14 )

Ns = [10,50,100]
ds = [10,50,100]

for d in ds:
    for N in Ns:
        ps = [rlsm(100, d, n_dates=N, n_paths=10000) for _ in range(10)]
        mean = np.round(np.mean(ps), decimals=2)
        std = np.round(np.std(ps), decimals=2)
        print("rlsm(", 100, ",", d, ",", "N =", N ,") =", mean, "(", std, ")")

# Output:
# rlsm( 100 , 10  , N = 10  ) = 32.73 ( 0.14 )
# rlsm( 100 , 10  , N = 50  ) = 31.54 ( 0.17 )
# rlsm( 100 , 10  , N = 100 ) = 31.04 ( 0.22 )
# rlsm( 100 , 50  , N = 10  ) = 52.28 ( 0.09 )
# rlsm( 100 , 50  , N = 50  ) = 50.62 ( 0.1  )
# rlsm( 100 , 50  , N = 100 ) = 50.15 ( 0.11 )
# rlsm( 100 , 100 , N = 10  ) = 60.55 ( 0.08 )
# rlsm( 100 , 100 , N = 50  ) = 58.77 ( 0.07 )
# rlsm( 100 , 100 , N = 100 ) = 58.26 ( 0.09 )

Ns = [10,50,100]
ds = [10,50,100]

for d in ds:
    for N in Ns:
        ps = [rfqi(100, d, n_dates=N, n_paths=10000,
                   hidden_nodes=min(d, 20), n_epochs=200,
                   verbose=True) for _ in range(10)]
        mean = np.round(np.mean(ps), decimals=2)
        std = np.round(np.std(ps), decimals=2)
        print("rfqi(", 100, ",", d, ",", "N =", N ,") =", mean, "(", std, ")")

# Output:
# rfqi( 100 , 10  , N = 10  ) = 33.63 ( 0.2  )
# rfqi( 100 , 10  , N = 50  ) = 34.05 ( 0.13 )
# rfqi( 100 , 10  , N = 100 ) = 34.23 ( 0.12 )
# rfqi( 100 , 50  , N = 10  ) = 53.57 ( 0.11 )
# rfqi( 100 , 50  , N = 50  ) = 53.88 ( 0.14 )
# rfqi( 100 , 50  , N = 100 ) = 54.01 ( 0.1  )
# rfqi( 100 , 100 , N = 10  ) = 61.59 ( 0.11 )
# rfqi( 100 , 100 , N = 50  ) = 61.93 ( 0.14 )
# rfqi( 100 , 100 , N = 100 ) = 62.06 ( 0.15 )
