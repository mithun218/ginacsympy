<p align="center">
<img src="https://github.com/mithun218/ginacsympy/blob/master/doc/html/img/logo.png?raw=true" width=400>
</p>

# GinacSympy: GinacSym Python Wrappers
A Cython frontend to the C++ library GinacSym (https://github.com/mithun218/ginacsym), a fast C++ symbolic manipulation library.

## Introduction
GinacSym (made on GiNaC, https://ginac.de) is a fast C++ symbolic manipulation library released under GPL v2 or above. It has many features:

- It is build on C and C++ libraries:
gmp,mpfr,flint,cln
- It  provides fast computations for multivariate polynomial operations (expansion and factorisation) using flint library, and
- symbolic  computations: solving linear or nonlinear systems of equations, simplifications, series, limit, differentiation, integration, Infinity, functions with dependend variable, real symbol,complex symbol,...
-  It has many special functions which are computed using flint library.
- Linear Algebra with numerical or symbolic coefficients.
- Indexed objects, tensors, symmetrization
- Clifford algebra
- Color algebra
- and many more
     
GinacSympy is an interface to this library. It is built with cython.
## Motivation

We are aware of the existence of the Sympy Python Library, a full computer algebra system. However, because sympy is a pure Python library (Sympy now depends on mpmath as an external library), symbolic manipulation of huge algebraic expressions is slow using it. Consequently, we have made an effort to create a computer algebra system that is both extremely fast in comparison to Sympy and quite simple to install, much like sympy.

## Installation
GinacSympy releases are available as wheel packages for Windows and Linux on [PyPi](https://pypi.org/project/ginacsympy). Install it using pip:

	python -m pip install -U pip
	python -m pip install -U ginacsympy

## Short usage

	>>>from ginacsympy import *
	>>>a,b,c,d=Ex('a,b,c,d')
	>>>expr=(a+b+c+d)**5
	>>>expr=expr.expand()
	>>>expr.factor()
	(a+b+c+d)**5
	>>>from ginacsympy_abc import *
	>>>(x**2*y+y*x+y*x*b+a*x+b*x**2*y).collect(lst([x,y]),True)
	x**2*(1+b)*y+x*(1+b)*y+a*x
	
For in-depth documentation [click here.](https://htmlpreview.github.io/?https://github.com/mithun218/ginacsympy/blob/master/doc/html/index.html)

	