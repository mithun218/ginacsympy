"""
This module exports all latin and greek letters as Symbols, so you can
conveniently do

    >>> from ginacsympy_abc import x, y

instead of the slightly more clunky-looking

    >>> from ginacsympy import Ex
    >>> x, y = Ex('x,y')

Caveats
=======

1. As of the time of writing this, the name ``I``, 
is colliding with name defined in GinacSympy. If you import this
from both ``ginacsympy_abc`` and ``ginacsympy``, the second import will "win".
This is an issue only for * imports, which should only be used for short-lived
code such as interactive sessions and throwaway scripts that do not survive
until the next GinacSympy upgrade, where ``GinacSympy`` may contain a different set of
names.

2. This module does not define symbol names on demand, i.e.
``from ginacsympy_abc import foo`` will be reported as an error because
``ginacsympy_abc`` does not contain the name ``foo``. To get a symbol named ``foo``,
you still need to use ``Ex('foo')``.
You can freely mix usage of ``ginacsympy_abc`` and ``Ex``, though
sticking with one and only one way to get the Ex does tend to make the code
more readable.

"""

from ginacsympy import Ex

##### Symbol definitions #####

# Implementation note: The easiest way to avoid typos in the Ex()
# parameter is to copy it from the left-hand side of the assignment.

a, b, c, d, e, f, g, h, i, j = Ex('a, b, c, d, e, f, g, h, i, j')
k, l, m, n, o, p, q, r, s, t = Ex('k, l, m, n, o, p, q, r, s, t')
u, v, w, x, y, z = Ex('u, v, w, x, y, z')

A, B, C, D, E, F, G, H, I, J = Ex('A, B, C, D, E, F, G, H, I, J')
K, L, M, N, O, P, Q, R, S, T = Ex('K, L, M, N, O, P, Q, R, S, T')
U, V, W, X, Y, Z = Ex('U, V, W, X, Y, Z')

alpha, beta, gamma, delta = Ex('\\alpha, \\beta, \\gamma, \\delta')
epsilon, zeta, eta, theta = Ex('\\epsilon, \\zeta, \\eta, \\theta')
iota, kappa, lamda, mu = Ex('\\iota, \\kappa, \\lamda, \\mu')
nu, xi, omicron, pi = Ex('\\nu, \\xi, \\omicron, \\pi')
rho, sigma, tau, upsilon = Ex('\\rho, \\sigma, \\tau, \\upsilon')
phi, chi, psi, omega = Ex('\\phi, \\chi, \\psi, \\omega')

