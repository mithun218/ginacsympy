import pytest
from ginacsympy import *
from ginacsympy_abc import *

@pytest.fixture
def symbols():
    x = Ex("x")
    y = Ex("y",complex)
    t = Ex("t",positive)
    s = Ex("s",real)
    return x, y, t, s


def test_lst_1():
    lst_v = lst([a,2,b])
    lst_v[2] = c
    assert lst_v == lst([a,2,c])

# def test_division_by_zero():
#     with pytest.raises(ZeroDivisionError):
#         core.divide(10, 0)



def test_symbol_creation(symbols):
    x, y, _, _ = symbols
    assert str(x) == "x"
    assert str(y) == "y"


def test_expand_polynomial(symbols):
    x, y, _, _ = symbols
    expr = (x + y)**2
    expanded = expand(expr)
    assert (expanded).is_equal(x**2+2*x*y+y**2)


def test_diff(symbols):
    x, _, _, _ = symbols
    expr = x**3 + 3*x
    d = diff(expr, x)
    assert (d).is_equal(3*x**2+3)

def test_trig_derivative(symbols):
    x, _, _, _ = symbols
    expr = sin(x)
    d = diff(expr, x)
    assert str(d) == "cos(x)"

def test_integrate_1(symbols):
    x, _, _, _ = symbols
    expr = x**2
    result = integrate(expr, x)
    assert (result).is_equal(div(1,3)*x**3)

def test_integrate_2():
    expr = alpha*exp(x)*sin(x)
    result = Integrate(expr,x).apply_partial_integration(2)
    assert (result).is_equal(alpha*(exp(x)*sin(x)-Integrate(exp(x)*sin(x),x))-alpha*cos(x)*exp(x))

def test_limit():
    expr = (sin(2*x)/x)**(x+1)
    result = limit(expr,x,0)
    assert str(result)=="2"

def test_simplify_1(symbols):
    x, _, _, _ = symbols
    expr = (x**2 - 1) / (x - 1)
    simplified = simplify(expr)
    assert (simplified).is_equal(x+1)

def test_simplify_2(symbols):
    _, _, t, _ = symbols
    expr = sqrt(t**2)
    assert str(expr) == "t"

def test_solve_quadratic():
    expr = a**2 - 4*a+4
    roots = solve(expr, a)
    assert roots[0].is_equal(lst([relational(a,2)]))

def test_series_expansion():
    expr = sin(a)
    ser = series(expr, a, 5).convert_to_poly(True)  # Taylor series up to a^4 term
    # Expect: 1*a+(-1/6)*a**3+Order(a**5)
    assert ser.is_equal(1*a+(-div(1,6))*a**3)

def test_laplace_transform(symbols):
    _, _, t, s = symbols
    expr = t**2
    F = laplace_transform(expr, t, s)
    assert (F).is_equal(2/s**3)

def test_inverse_laplace_transform(symbols):
    _, _, t, s = symbols
    F = 1/(s**2 + 1)
    f = inverse_laplace_transform(F, s, t)
    assert str(f) == "sin(t)"

def test_apart_1(symbols):
    expr = (x**2 + x + 1) / (x*(x+1))
    aparted = apart(expr, x)
    assert aparted.is_equal(1-(1+x)**(-1)+x**(-1))

def test_apart_2(symbols):
    expr = (x**2 + x + 1) / (x*(x+1))
    aparted = apart_with_steps(expr, x)
    assert aparted.is_equal(1-(1+x)**(-1)+x**(-1))

