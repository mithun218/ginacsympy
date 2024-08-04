# GinacSym: a fast C++ symbolic manipulation library

GinacSym is a fast C++ symbolic manipulation library. It has been made on GiNaC c++ library (https://ginac.de). It has many additional features in compared to GiNaC library, some are functions f(x), Infinity, inert differential Diff, inert integration Integration, integrate, solve of system of nonlinear polynomial equations, simplify of algebraic expressions, many new mathematical functions, powerful factor of polynomial expressions using flint c library etc. The main aim of this project ginacsym  is to become a complete computer algebra system by  adding many new features to ginac library, and creating its a python wrapper GinacSympy (https://github.com/mithun218/ginacsympy). GinacSympy is also a beautiful project which is a Cython frontend to ginacsym, and in jupyter notebook we can make all mathematical computations with mathjax  equation  rendering.

## Main differences between GiNaC and GinacSym
There are two main differences between GiNaC and GinacSym:

1. In GiNaC, symbols are declared by *symbol, realsymbol, possymbol* class, and mathematical expressions from string are parsed by *parser* class. 

	In GinacSym, symbols and mathematical expressions from string are created by *generator* class, and all generated symbols are collected in *generator* class. In this regard, following code shows a brief comparison between GiNaC code and GinacSym code:
	
	**GiNaC code:** 
	
		symbol x("x");
		realsymbol y("y");
		possymbol z("z");
		parser reader;
		ex expr=reader("x^2+y^2");
		
	Corresponding **GinacSym code:**
	
		ex x=generator.symGenerator("x",symbol_assumptions::symbol);
		ex y=generator.symGenerator("y",symbol_assumptions::realsymbol);
		ex z=generator.symGenerator("z",symbol_assumptions::possymbol);
		ex expr=generator.exGenerator("x^2+y^2");

2. In GiNaC, the map_function class declares a virtual function call operator (()) that we can overload. But in GinacSym, the map_function class declares a virtual function ***expr_visitior()*** (instead of call operator) that we can overload.

	**GiNaC code:**

		struct map_rem_quad : public map_function {
		 ex var;
		 map_rem_quad(const ex & var_) : var(var_) {}
		 ex operator()(const ex & e)
		 {
		  if (is_a<add>(e) || is_a<mul>(e))
		   return e.map(*this);
		  else if (is_a<power>(e) &&
		          e.op(0).is_equal(var) && e.op(1).info(info_flags::even))
		   return 0;
		  else
		   return e;
		 }
		};


	Corresponding **GinacSym code:**

		struct map_rem_quad : public map_function {
		 ex var;
		 map_rem_quad(const ex & var_) : var(var_) {}
		 ex expr_visitor(const ex & e)
		 {
		  if (is_a<add>(e) || is_a<mul>(e))
		   return e.map(*this);
		  else if (is_a<power>(e) &&
		          e.op(0).is_equal(var) && e.op(1).info(info_flags::even))
		   return 0;
		  else
		   return e;
		 }
		};

	We have made this change because it is impossible to wrap virtual function call operator in the python wrapper GinacSympy, but we can easily wrap ***expr_visitior()*** in GinacSympy.

## Additional Features
In contrast to GiNaC, GinacSym has the following additional features:

- functions dependent on variable (e.g. f(x))
- Infinity
- Diff-> inert form of diff
- integrate
- Integrate-> inert form of integrate
- solve-> system of nonlinear polynomial equations
- simplify-> simplification of algebraic expressions
- factor-> powerful algorithm for factorisations of multivariate polynomial equations using flint library
- expand-> powerful algorithm for expansions of multivariate polynomial equations using flint library
- fast computations of mathematical special functions using flint library-> 
integral functions (sinIntegral, cosIntegral, sinhIntegral, coshIntegral, logIntegral, expIntegralE, expIntegralEi), Chebyshev T polynomial, Chebyshev U polynomial, Legendre P polynomial, Legendre Q (second kind), Hermite polynomial, Gegenbauer (ultraspherical) polynomial, Bessel J function (first kind), Bessel Y function (second kind), Modified Bessel functions (first kind), Modified Bessel functions (second kind).

## Documentation
For in-depth documentation please see at http://htmlpreview.github.io/?https://github.com/mithun218/ginacsym/doc/html/index.html

## To do

We intend to enhance GinacSym with numerous new features in order to build a complete computer algebra system. Initially, the following has been identified to fulfil our goal:
- Adding more special functions using flint library.
- Adding differential equation solver
- Series solutions of differential equations
- Laplace transform
- Fourier transform
- Fourier serries
- Dirac delta, Kronecker delta functions
- and many more.

## Install
To install GinacSym, we require C library  [flint >= 3.0.0](https://flintlib.org) and c++ library [CLN >= 1.3.4](http://www.ginac.de/CLN/). After installing these dependencies, we can install GinacSym using the following commands:
```        
        $ ./configure
        $ make
        $ make check
        $ make install