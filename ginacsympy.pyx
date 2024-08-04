# distutils: sources = imap_function.cpp
# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list as cpplist
from libcpp.set cimport set as cppset
from libcpp.map cimport map as cppmap
from libcpp.pair cimport pair
from libcpp cimport bool
#from cpython.array cimport array
from cython.operator cimport dereference as deref, preincrement as inc, predecrement as dec
cimport cpython.ref as cpy_ref
from cpython.ref cimport PyObject

#from IPython.display import display, Math

#ginacsympy version number
import ginacsympy_version
__version__ = ginacsympy_version.__version__

#cdef extern from "string" namespace "std":
#    cdef cppclass string:
#        string(const string& inp)
#        string(const char* inp)
#        char* c_str()
########################## pyd ############################
cdef extern from "ginacsym/ptr.h" namespace "ginacsym":
    cdef cppclass refcounted:
        pass
cdef extern from "ginacsym/basic.h" namespace "ginacsym":
    ctypedef vector[ex] exvector
    ctypedef cppmap[ex, ex, ex_is_less] exmap
    ctypedef cppset[ex] exset
    ex _dynallocate "ginacsym::dynallocate"[T]()
    T _dynallocate1 "ginacsym::dynallocate"[T](cpplist[ex] lexpr)
    cdef cppclass basic(refcounted):
        const char *class_name()
#    cdef cppclass map_function:
#        pass

cdef extern from "ginacsym/expairseq.h" namespace "ginacsym":
    cdef cppclass expairseq(basic):
        pass
    cdef cppclass expair:
        expair(const ex & r, const ex & c)
    ctypedef vector[expair] epvector

cdef extern from "ginacsym/pseries.h" namespace "ginacsym":
    cdef cppclass _pseries "ginacsym::pseries"(basic):
        _pseries()
        _pseries(const ex &rel_, const epvector &ops_)
#        /** Get the expansion variable. */
        ex get_var() except +

#	/** Get the expansion point. */
        ex get_point()  except +

#        /** Convert the pseries object to an ordinary polynomial.
#         *
#         *  @param no_order flag: discard higher order terms */
        ex convert_to_poly(bool no_order) except +

#        /** Check whether series is compatible to another series (expansion
#         *  variable and point are the same. */
        bool is_compatible_to(const _pseries &other) except +
#        /** Check whether series has the value zero. */
        bool is_zero() except +

#        /** Returns true if there is no order term, i.e. the series terminates and
#         *  false otherwise. */
        bool is_terminating() except +

#        /** Get coefficients and exponents. */
        ex coeffop(size_t i) except +
        ex exponop(size_t i) except +

        ex add_series(const _pseries &other) except +
        ex mul_const(const _numeric &other) except +
        ex mul_series(const _pseries &other) except +
        ex power_const(const _numeric &p, int deg) except +
        _pseries shift_exponents(int deg) except +


cdef extern from "ginacsym/symbol.h" namespace "ginacsym":
    cdef cppclass _symbol "ginacsym::symbol"(basic):
        _symbol()
        _symbol(const string & initname) except +
        _symbol(const string & initname, const string & texname) except +
        ex eval() except +

cdef extern from "ginacsym/lst.h" namespace "ginacsym":
    cdef cppclass _lst "ginacsym::lst":
        pass

cdef extern from "ginacsym/relational.h" namespace "ginacsym":
    cdef enum _operators:
        _equal "ginacsym::relational::equal"
        _not_equal "ginacsym::relational::not_equal"
        _less "ginacsym::relational::less"
        _less_or_equal "ginacsym::relational::less_or_equal"
        _greater "ginacsym::relational::greater"
        _greater_or_equal "ginacsym::relational::greater_or_equal"

    cdef cppclass _relational "ginacsym::relational"(basic):
        _relational()
        _relational(const ex & lhs, const ex & rhs, _operators oper)#=equal)

cdef extern from "ginacsym/lst.h" namespace "ginacsym":
    cdef _lst cpplist_to__lst "ginacsym::lst"(cpplist[ex] inp)
    cdef ex cpplist_to_ex "ginacsym::lst"(cpplist[ex] inp)
    cdef ex _lst_to_ex "ginacsym::lst"(_lst inp)

cdef extern from "ginacsym/ex.h" namespace "ginacsym":
    cdef cppclass ex:
        # constructors
        ex()
        ex(int i)
        ex(double d)


        # iterators
        _const_iterator begin() except +
        _const_iterator end() except +
        _const_preorder_iterator preorder_begin() except +
        _const_preorder_iterator preorder_end() except +
        _const_postorder_iterator postorder_begin() except +
        _const_postorder_iterator postorder_end() except +

        # evaluation
        ex eval() except +
        ex evalf() except +
        ex evalm() except +
        ex eval_ncmul(const exvector & v) except +
        ex eval_integ() except +

        # info
        bool info(unsigned inf) except +

        # operand access
        size_t nops() except +
        ex op(size_t i) except +
#        ex operator[](const ex & index) except +
#        ex operator[](size_t i) except +
        ex & let_op(size_t i) except +
        ex & operator[](const ex & index) except +
        ex & operator[](size_t i) except +
        ex lhs() except +
        ex rhs() except +

        # function for complex expressions
        ex conjugate() except +
        ex real_part() except +
        ex imag_part() except +

        # pattern matching
        bool has(const ex & pattern, unsigned options) except +#= 0)
        bool find(const ex & pattern, exset& found) except +
        bool match(const ex & pattern) except +
        bool match(const ex & pattern, exmap & repls) except +

        # substitutions
        ex subs(const exmap & m, unsigned options)  except +
        ex subs(const cpplist[ex] & ls, const cpplist[ex] & lr, unsigned options) except +
        ex subs(const ex & e, unsigned options)

        # function mapping
        ex map(imap_function & f) except +
        ex map(ex (*f)(const ex & e))

        # visitors and tree traversal
#        void accept(visitor & v)
#        void traverse_preorder(visitor & v)
#        void traverse_postorder(visitor & v)
#        void traverse(visitor & v)

        # degree/coeff
        bool is_polynomial(const ex & vars) except +
        int degree(const ex & s) except +
        int ldegree(const ex & s) except +
        ex coeff(const ex & s, int n) except +
        ex lcoeff(const ex & s) except +
        ex tcoeff(const ex & s) except +

        # expand/collect
        ex power_expand(unsigned options) except +
        ex expand(unsigned options) except +
        ex collect(const ex & s, bool distributed) except +

        # differentiation and series expansion
        ex diff(const _symbol& s, unsigned nth) except +
        ex series(const ex & r, int order, unsigned options) except +

        # rational functions
        ex normal() except +
        ex to_rational(exmap & repl) except +
        ex to_polynomial(exmap & repl) except +
        ex numer() except +
        ex denom() except +
        ex numer_denom() except +

        # polynomial algorithms
        ex unit(const ex &x) except +
        ex content(const ex &x) except +
        _numeric integer_content() except +
        ex primpart(const ex &x) except +
        ex primpart(const ex &x, const ex &cont) except +
        void unitcontprim(const ex &x, ex &u, ex &c, ex &p) except +
        ex smod(const _numeric &xi) except +
        _numeric max_coefficient() except +

        # indexed objects
        exvector get_free_indices() except +
        ex simplify_indexed(unsigned options) except +
        ex simplify_indexed(const _scalar_products & sp, unsigned options) except +

        # comparison
        int compare(const ex & other) except +
        bool is_equal(const ex & other) except +
        bool is_zero() except +
        bool is_zero_matrix() except +

        # symmetry
        ex symmetrize() except +
        #ex symmetrize(const lst & l) except +
        ex antisymmetrize() except +
        #ex antisymmetrize(const lst & l) except +
        ex symmetrize_cyclic() except +
        #ex symmetrize_cyclic(const lst & l) except +

        # noncommutativity
        unsigned return_type() except +
        _return_type_t return_type_tinfo() except +

        unsigned gethash() except +

    cdef cppclass _const_iterator "ginacsym::const_iterator":
        ex operator*() except + # call by deref
        _const_iterator &operator++() except + #call by inc
        _const_iterator &operator--() except + #call by dec
        bool operator!=(const _const_iterator &other) except +

    cdef cppclass _const_preorder_iterator "ginacsym::const_preorder_iterator":
        ex operator*() except + # call by deref
        _const_preorder_iterator &operator++() except + #call by inc
        _const_preorder_iterator &operator--() except + #call by dec
        bool operator!=(const _const_preorder_iterator &other) except +

    cdef cppclass _const_postorder_iterator "ginacsym::const_postorder_iterator":
        ex operator*() except + # call by deref
        _const_postorder_iterator &operator++() except + #call by inc
        _const_postorder_iterator &operator--() except + #call by dec
        bool operator!=(const _const_postorder_iterator &other) except +
#    // utility functions
    cdef cppclass ex_is_less:
        pass
    bool is_a[T](const ex &obj) except +
    T &ex_to[T](const ex &e) except +

    bool operator!=(const ex & lh, const ex & rh)  except +
    bool operator<(const ex & lh, const ex & rh)  except +
    bool operator<=(const ex & lh, const ex & rh)  except +
    bool operator>(const ex & lh, const ex & rh)  except +
    bool operator>=(const ex & lh, const ex & rh)  except +

cdef extern from "ginacsym/factor.h" namespace "ginacsym":
    ex _factor "ginacsym::factor"(const ex& poly, unsigned options) except +
    ex expandflint(const ex& e, unsigned options) except +

cdef extern from "ginacsym/operators.h" namespace "ginacsym":
    ex operator+(const ex & lh, const ex & rh) except +
    ex operator-(const ex & lh, const ex & rh) except +
    ex operator*(const ex & lh, const ex & rh) except +
    ex operator/(const ex & lh, const ex & rh) except +
    ex operator+(const ex & lh) except +
    ex operator-(const ex & lh) except +
cdef extern from "ginacsym/add.h" namespace "ginacsym":
    cdef cppclass _add "ginacsym::add"(expairseq):
        _add()
        _add(const ex & lh, const ex & rh) except +
        _add(const exvector& v) except +
        ex eval() except +
cdef extern from "ginacsym/mul.h" namespace "ginacsym":
    cdef cppclass _mul "ginacsym::mul"(expairseq):
        _mul()
        _mul(const ex & lh, const ex & rh) except +
        _mul(const exvector& v) except +
        ex eval() except +
cdef extern from "ginacsym/power.h" namespace "ginacsym":
    cdef cppclass _power "ginacsym::power"(expairseq):
        _power()
        _power(const ex & basic, const ex & exponent) except +
        ex eval() except +
    ex _pow "ginacsym::pow"(const ex & b, const ex & e) except +

cdef extern from "ginacsym/wildcard.h" namespace "ginacsym":
    cdef cppclass _wildcard "ginacsym::wildcard"(basic):
#        /** Construct wildcard with specified label. */
        _wildcard()
        _wildcard(unsigned label) except +
        unsigned get_label() except +
        ex eval() except +
    cdef ex _wild "ginacsym::wild"(unsigned label)
    ## Check whether x has a wildcard anywhere as a subexpression. ##
    cdef bool _haswild "ginacsym::haswild"(const ex & x)

cdef extern from "ginacsym/parse_context.h" namespace "ginacsym":
    ctypedef cppmap[string, ex] symtab

cdef extern from "ginacsym/parser.h" namespace "ginacsym":
    cdef cppclass parser:
        pass

cdef extern from "ginacsym/flags.h" namespace "ginacsym":

    cdef enum _symbol_assumptions:
        _symbol_assumptions_symbol "ginacsym::symbol_assumptions::symbol"
        _realsymbol "ginacsym::symbol_assumptions::realsymbol"
        _possymbol "ginacsym::symbol_assumptions::possymbol"

    cdef enum _expand_options:
        _expand_indexed "ginacsym::expand_options::expand_indexed"
        _expand_function_args "ginacsym::expand_options::expand_function_args"
        _expand_rename_idx "ginacsym::expand_options::expand_rename_idx"
        _expand_transcendental "ginacsym::expand_options::expand_transcendental"

    cdef enum _has_options:
        _has_algebraic "ginacsym::has_options::algebraic"

    cdef enum _subs_options:
        _no_pattern "ginacsym::subs_options::no_pattern"
        _subs_no_pattern "ginacsym::subs_options::subs_no_pattern"
        _algebraic "ginacsym::subs_options::algebraic"
        _subs_algebraic "ginacsym::subs_options::subs_algebraic"
        _no_index_renaming "ginacsym::subs_options::no_index_renaming"
        _really_subs_idx "ginacsym::subs_options::really_subs_idx"

    cdef enum _series_options:
        _suppress_branchcut "ginacsym::series_options::suppress_branchcut"

    cdef enum _factor_options:
        _polynomial "ginacsym::factor_options::polynomial"
        _all "ginacsym::factor_options::all"

    cdef enum _determinant_algo:
        _determinant_algo_automatic "ginacsym::determinant_algo::automatic"
        _determinant_algo_gauss "ginacsym::determinant_algo::gauss"
        _determinant_algo_divfree "ginacsym::determinant_algo::divfree"
        _determinant_algo_laplace "ginacsym::determinant_algo::laplace"
        _determinant_algo_bareiss "ginacsym::determinant_algo::bareiss"

    cdef enum _solve_algo:
        _solve_algo_automatic "ginacsym::solve_algo::automatic"
        _solve_algo_gauss "ginacsym::solve_algo::gauss"
        _solve_algo_divfree "ginacsym::solve_algo::divfree"
        _solve_algo_bareiss "ginacsym::solve_algo::bareiss"
        _solve_algo_markowitz "ginacsym::solve_algo::markowitz"

#    /** Possible attributes an object can have. */
    cdef enum _info_flags:
        #        // answered by class numeric, add, mul, function and symbols/constants in particular domains
        _info_flags_numeric "ginacsym::info_flags::numeric"
        _real "ginacsym::info_flags::real"
        _rational "ginacsym::info_flags::rational"
        _integer "ginacsym::info_flags::integer"
        _crational "ginacsym::info_flags::crational"
        _cinteger "ginacsym::info_flags::cinteger"
        _positive "ginacsym::info_flags::positive"
        _negative "ginacsym::info_flags::negative"
        _nonnegative "ginacsym::info_flags::nonnegative"
        _posint "ginacsym::info_flags::posint"
        _negint "ginacsym::info_flags::negint"
        _nonnegint "ginacsym::info_flags::nonnegint"
        _even "ginacsym::info_flags::even"
        _odd "ginacsym::info_flags::odd"
        _prime "ginacsym::info_flags::prime"

        #        // answered by class relation
        _relation "ginacsym::info_flags::relation"
        _relation_equal "ginacsym::info_flags::relation_equal"
        _relation_not_equal "ginacsym::info_flags::relation_not_equal"
        _relation_less "ginacsym::info_flags::relation_less"
        _relation_less_or_equal "ginacsym::info_flags::relation_less_or_equal"
        _relation_greater "ginacsym::info_flags::relation_greater"
        _relation_greater_or_equal "ginacsym::info_flags::relation_greater_or_equal"

        #        // answered by class symbol
        _info_flags_symbol "ginacsym::info_flags::symbol"

        #        // answered by class lst
        _list "ginacsym::info_flags::list"

        #        // answered by class exprseq
        _info_flags_exprseq "ginacsym::info_flags::exprseq"

        #        // answered by classes numeric, symbol, add, mul, power
        _info_flags_polynomial "ginacsym::info_flags::polynomial"
        _integer_polynomial "ginacsym::info_flags::integer_polynomial"
        _cinteger_polynomial "ginacsym::info_flags::cinteger_polynomial"
        _rational_polynomial "ginacsym::info_flags::rational_polynomial"
        _crational_polynomial "ginacsym::info_flags::crational_polynomial"
        _rational_function "ginacsym::info_flags::rational_function"

        #        // answered by class indexed
        _info_flags_indexed "ginacsym::info_flags::indexed"      # class can carry indices
        _has_indices "ginacsym::info_flags::has_indices"  # object has at least one index

        #        // answered by class idx
        _info_flags_idx "ginacsym::info_flags::idx"

        #        // answered by classes numeric, symbol, add, mul, power
        _expanded "ginacsym::info_flags::expanded"

        #        // is meaningful for mul only
        _indefinite "ginacsym::info_flags::indefinite"

    cdef enum _return_types:
        _commutative "ginacsym::return_types::commutative"
        _noncommutative "ginacsym::return_types::noncommutative"
        _noncommutative_composite "ginacsym::return_types::noncommutative_composite"

cdef extern from "ginacsym/numeric.h" namespace "ginacsym":
    cdef cppclass _numeric "ginacsym::numeric"(basic):
        _numeric()
        _numeric(int i)
        _numeric(const char *)
        ex eval() except +
        int csgn()except +
        int compare(const _numeric &other)except +
        bool is_equal(const _numeric &other)except +
        bool is_zero()except +
        bool is_positive()except +
        bool is_negative()except +
        bool is_integer()except +
        bool is_pos_integer()except +
        bool is_nonneg_integer()except +
        bool is_even()except +
        bool is_odd()except +
        bool is_prime()except +
        bool is_rational()except +
        bool is_real()except +
        bool is_cinteger()except +
        bool is_crational()except +

        long to_long() const
        double to_double() const;

    ex _numeric_to_ex(const _numeric& n)
    _numeric string_to__numeric(const string& inp)
    _numeric _I "ginacsym::I"

cdef extern from "ginacsym/number.h" namespace "ginacsym":
#    //the digits of n in base b
    string _base_form "ginacsym::base_form"(const ex& n, const int &b) except +

#    //test whether m is divisible by n
    bool _divisible "ginacsym::divisible"(const ex& m, const ex& n) except +

#    //next smallest prime above n
    ex _next_prime "ginacsym::next_prime"(const ex& n) except +

#    //random prime generator less than and equal to n
    ex _random_prime "ginacsym::random_prime"(const ex& n) except +

cdef extern from "ginacsym/constant.h" namespace "ginacsym":
    cdef cppclass _constant "ginacsym::constant"(basic):
        pass
    ex _Pi "ginacsym::Pi"
    ex _Euler "ginacsym::Euler"
    ex _Catalan "ginacsym::Catalan"

cdef extern from "ginacsym/normal.h" namespace "ginacsym":
#    /**
#     * Flags to control the behavior of gcd() and friends
#     */
    cdef enum _gcd_options:
#        /**
#         * Usually ginacsym tries heuristic GCD first, because typically
#         * it's much faster than anything else. Even if heuristic
#         * algorithm fails, the overhead is negligible w.r.t. cost
#         * of computing the GCD by some other method. However, some
#         * people dislike it, so here's a flag which tells ginacsym
#         * to NOT use the heuristic algorithm.
#         */
        _no_heur_gcd "ginacsym::gcd_options::no_heur_gcd"
#        /**
#         * ginacsym tries to avoid expanding expressions when computing
#         * GCDs. This is a good idea, but some people dislike it.
#         * Hence the flag to disable special handling of partially
#         * factored polynomials. DON'T SET THIS unless you *really*
#         * know what are you doing!
#         */
        _no_part_factored "ginacsym::gcd_options::no_part_factored"
#        /**
#         * By default ginacsym uses modular GCD algorithm. Typically
#         * it's much faster than PRS (pseudo remainder sequence)
#         * algorithm. This flag forces ginacsym to use PRS algorithm
#         */
        _use_sr_gcd "ginacsym::gcd_options::use_sr_gcd"


#    // Quotient q(x) of polynomials a(x) and b(x) in Q[x], so that a(x)=b(x)*q(x)+r(x)
    ex _quo "ginacsym::quo"(const ex &a, const ex &b, const ex &x, bool check_args) except +

#    // Remainder r(x) of polynomials a(x) and b(x) in Q[x], so that a(x)=b(x)*q(x)+r(x)
    ex _rem "ginacsym::rem"(const ex &a, const ex &b, const ex &x, bool check_args) except +

#    // Decompose rational function a(x)=N(x)/D(x) into Q(x)+R(x)/D(x) with degree(R, x) < degree(D, x)
    ex _decomp_rational "ginacsym::decomp_rational"(const ex &a, const ex &x) except +

#    // Pseudo-remainder of polynomials a(x) and b(x) in Q[x]
    ex _prem "ginacsym::prem"(const ex &a, const ex &b, const ex &x, bool check_args) except +

#    // Pseudo-remainder of polynomials a(x) and b(x) in Q[x]
    ex _sprem "ginacsym::sprem"(const ex &a, const ex &b, const ex &x, bool check_args) except +

#    // Exact polynomial division of a(X) by b(X) in Q[X] (quotient returned in q), returns false when exact division fails
    bool _divide "ginacsym::divide"(const ex &a, const ex &b, ex &q, bool check_args) except +

#    // Polynomial GCD in Z[X], cofactors are returned in ca and cb, if desired
    ex _gcd "ginacsym::gcd"(const ex &a, const ex &b, ex *ca , ex *cb, bool check_args, unsigned options) except +

#    // Polynomial LCM in Z[X]
    ex _lcm "ginacsym::lcm"(const ex &a, const ex &b, bool check_args) except +

#    // Square-free factorization of a polynomial a(x)
    ex _sqrfree "ginacsym::sqrfree"(const ex &a, const _lst &l) except +

#    // Square-free partial fraction decomposition of a rational function a(x)
    ex _sqrfree_parfrac "ginacsym::sqrfree_parfrac"(const ex & a, const _symbol & x) except +

#    /** Compute square-free partial fraction decomposition of rational function **/
    ex _apart "ginacsym::apart"(const ex & a, const _symbol & x)  except +

##    // Collect common factors in sums.
#    ex _collect_common_factors "ginacsym::collect_common_factors"(const ex & e) except +

#    // Resultant of two polynomials e1,e2 with respect to symbol s.
    ex _resultant "ginacsym::resultant"(const ex & e1, const ex & e2, const ex & s) except +


cdef extern from "imap_functions.h" namespace "ginacsym":
    cdef cppclass imap_function:
        imap_function(cpy_ref.PyObject *obj) except +
        ex expr_visitor(const ex & e) except +

cdef extern from "ginacsym/registrar.h" namespace "ginacsym":
    cdef cppclass _return_type_t "ginacsym::return_type_t":
        _return_type_t()
        bool operator<(const _return_type_t& other)
        bool operator==(const _return_type_t& other)
        bool operator!=(const _return_type_t& other)

cdef extern from "ginacsym/matrix.h" namespace "ginacsym":
    cdef cppclass _matrix "ginacsym::matrix"(basic):
        #matrix(cpplist[cpplist[ex]] l)
#        size_t nops() except +
#        ex op(size_t i) except +
#        ex & let_op(size_t i) except +
        ex evalm() except +
        ex eval_indexed(const basic & i) except +
        ex add_indexed(const ex & self, const ex & other) except +
        ex scalar_mul_indexed(const ex & self, const _numeric & other) except +
        #bool contract_with(exvector::iterator self, exvector::iterator other, exvector & v)
        ex conjugate() except +
        ex real_part() except +
        ex imag_part() except +
        unsigned rows() except +      #/ Get number of rows.
        unsigned cols() except +      #/ Get number of columns.
        _matrix add(const _matrix & other) except +
        _matrix sub(const _matrix & other) except +
        _matrix mul(const _matrix & other) except +
        _matrix mul(const _numeric & other) except +
        _matrix mul_scalar(const ex & other) except +
        _matrix pow(const ex & expn) except +
        ex & operator() (unsigned ro, unsigned co) except +
        _matrix & set(unsigned ro, unsigned co, const ex & value) except +
        _matrix transpose() except +
        ex determinant(unsigned algo) except +
        ex trace() except +
        ex charpoly(const ex & lambda1) except +
        _matrix inverse() except +
        _matrix inverse(unsigned algo) except +
        _matrix solve(const _matrix & vars, const _matrix & rhs, unsigned algo) except +
        unsigned rank() except +
        unsigned rank(unsigned solve_algo) except +
        bool is_zero_matrix() except +

#    /** Convert list of diagonal elements to matrix. */
    ex _diag_matrix "ginacsym::diag_matrix"(const _lst & l) except +
#    extern ex diag_matrix(std::initializer_list<ex> l);

#    /** Create an r times c unit matrix. */
    ex _unit_matrix "ginacsym::unit_matrix"(unsigned r, unsigned c) except +

#    /** Create a x times x unit matrix. */
    ex _unit_matrix "ginacsym::unit_matrix"(unsigned x) except +

#    /** Create an r times c matrix of newly generated symbols consisting of the
#     *  given base name plus the numeric row/column position of each element.
#     *  The base name for LaTeX output is specified separately. */
    ex _symbolic_matrix "ginacsym::symbolic_matrix"(unsigned r, unsigned c, const string & base_name, const string & tex_base_name)  except +

#    /** Return the reduced matrix that is formed by deleting the rth row and cth
#     *  column of matrix m. The determinant of the result is the Minor r, c. */
    ex reduced_matrix(const _matrix& m, unsigned r, unsigned c) except +

#    /** Return the nr times nc submatrix starting at position r, c of matrix m. */
    ex sub_matrix(const _matrix&m, unsigned r, unsigned nr, unsigned c, unsigned nc) except +

#    /** Create an r times c matrix of newly generated symbols consisting of the
#     *  given base name plus the numeric row/column position of each element. */
    ex _symbolic_matrix "ginacsym::symbolic_matrix"(unsigned r, unsigned c, const string & base_name) except +

cdef extern from "ginacsym/ncmul.h" namespace "ginacsym":
    cdef cppclass _ncmul "ginacsym::ncmul":
        _ncmul()
        _ncmul(const exvector & v) except +

        const exvector & get_factors() except +


cdef extern from "ginacsym/symmetry.h" namespace "ginacsym":
    ctypedef enum _symmetry_type "ginacsym::symmetry::symmetry_type":
            _none   "ginacsym::symmetry::symmetry_type::none"       #/**< no symmetry properties */
            _symmetric "ginacsym::symmetry::symmetry_type::symmetric"     #/**< totally symmetric */
            _antisymmetric "ginacsym::symmetry::symmetry_type::antisymmetric"#/**< totally antisymmetric */
            _cyclic  "ginacsym::symmetry::symmetry_type::cyclic"       #/**< cyclic symmetry */

    cdef cppclass _symmetry "ginacsym::symmetry"(basic):
#        ctypedef enum __symmetry_type "ginacsym::symmetry_type":
#            none          #/**< no symmetry properties */
#            symmetric     #/**< totally symmetric */
#            antisymmetric #/**< totally antisymmetric */
#            cyclic         #/**< cyclic symmetry */

        _symmetry()
#	// other constructors
#	/** Create leaf node that represents one index. */
        _symmetry(unsigned i);

#	/** Create node with two children. */
        _symmetry(_symmetry_type t, const _symmetry &c1, const _symmetry &c2);

#	/** Get symmetry type. */
        _symmetry_type get_type() except +

#	/** Set symmetry type. */
        void set_type(_symmetry_type t) except +

#	/** Add child node, check index sets for consistency. */
        _symmetry &add(const _symmetry &c) except +

#	/** Verify that all indices of this node are in the range [0..n-1].
#	 *  This function throws an exception if the verification fails.
#	 *  If the top node has a type != none and no children, add all indices
#	 *  in the range [0..n-1] as children. */
        void validate(unsigned n) except +

#	/** Check whether this node actually represents any kind of symmetry. */
        bool has_symmetry() except +
#	/** Check whether this node involves anything non symmetric. */
        bool has_nonsymmetric() except +
#	/** Check whether this node involves a cyclic symmetry. */
        bool has_cyclic() except +

#    // global functions
    _symmetry _sy_none "ginacsym::sy_none"() except +
    _symmetry _sy_none "ginacsym::sy_none"(const _symmetry &c1, const _symmetry &c2) except +
    _symmetry _sy_none "ginacsym::sy_none"(const _symmetry &c1, const _symmetry &c2, const _symmetry &c3) except +
    _symmetry _sy_none "ginacsym::sy_none"(const _symmetry &c1, const _symmetry &c2, const _symmetry &c3, const _symmetry &c4) except +

    _symmetry _sy_symm "ginacsym::sy_symm"() except +
    _symmetry _sy_symm "ginacsym::sy_symm"(const _symmetry &c1, const _symmetry &c2) except +
    _symmetry _sy_symm "ginacsym::sy_symm"(const _symmetry &c1, const _symmetry &c2, const _symmetry &c3) except +
    _symmetry _sy_symm "ginacsym::sy_symm"(const _symmetry &c1, const _symmetry &c2, const _symmetry &c3, const _symmetry &c4) except +

    _symmetry _sy_anti "ginacsym::sy_anti"() except +
    _symmetry _sy_anti "ginacsym::sy_anti"(const _symmetry &c1, const _symmetry &c2) except +
    _symmetry _sy_anti "ginacsym::sy_anti"(const _symmetry &c1, const _symmetry &c2, const _symmetry &c3) except +
    _symmetry _sy_anti "ginacsym::sy_anti"(const _symmetry &c1, const _symmetry &c2, const _symmetry &c3, const _symmetry &c4) except +

    _symmetry _sy_cycl "ginacsym::sy_cycl"() except +
    _symmetry _sy_cycl "ginacsym::sy_cycl"(const _symmetry &c1, const _symmetry &c2) except +
    _symmetry _sy_cycl "ginacsym::sy_cycl"(const _symmetry &c1, const _symmetry &c2, const _symmetry &c3) except +
    _symmetry _sy_cycl "ginacsym::sy_cycl"(const _symmetry &c1, const _symmetry &c2, const _symmetry &c3, const _symmetry &c4) except +

#    // These return references to preallocated common symmetries (similar to
#    // the numeric flyweights) except +.
    const _symmetry & _not_symmetric "ginacsym::not_symmetric"() except +
    const _symmetry & _symmetric2 "ginacsym::symmetric2"() except +
    const _symmetry & _symmetric3 "ginacsym::symmetric3"() except +
    const _symmetry & _symmetric4 "ginacsym::symmetric4"() except +
    const _symmetry & _antisymmetric2 "ginacsym::antisymmetric2"() except +
    const _symmetry & _antisymmetric3 "ginacsym::antisymmetric3"() except +
    const _symmetry & _antisymmetric4 "ginacsym::antisymmetric4"() except +

#    /** Canonicalize the order of elements of an expression vector, according to
#     *  the symmetry properties defined in a symmetry tree.
#     *
#     *  @param v Start of expression vector
#     *  @param symm Root node of symmetry tree
#     *  @return the overall sign introduced by the reordering (+1, -1 or 0)
#     *          or numeric_limits<int>::max() if nothing changed */
    #int canonicalize(exvector::iterator v, const symmetry &symm)

#    /** Symmetrize expression over a set of objects (symbols, indices). */
    #ex symmetrize(const ex & e, exvector::const_iterator first, exvector::const_iterator last)

#    /** Symmetrize expression over a set of objects (symbols, indices). */
    ex _symmetrize "ginacsym::symmetrize"(const ex & e, const exvector & v) except +

#    /** Antisymmetrize expression over a set of objects (symbols, indices). */
    #ex antisymmetrize(const ex & e, exvector::const_iterator first, exvector::const_iterator last)

#    /** Antisymmetrize expression over a set of objects (symbols, indices). */
    ex _antisymmetrize "ginacsym::antisymmetrize"(const ex & e, const exvector & v) except +

#    /** Symmetrize expression by cyclic permutation over a set of objects
#     *  (symbols, indices). */
    #ex symmetrize_cyclic(const ex & e, exvector::const_iterator first, exvector::const_iterator last)

#    /** Symmetrize expression by cyclic permutation over a set of objects
#     *  (symbols, indices). */
    ex _symmetrize_cyclic "ginacsym::symmetrize_cyclic"(const ex & e, const exvector & v) except +


cdef extern from "ginacsym/idx.h" namespace "ginacsym":
    cdef cppclass _idx "ginacsym::idx"(basic):
        _idx()
#        /** Construct index with given value and dimension.
#	 *
#	 *  @param v Value of index (numeric or symbolic)
#	 *  @param dim Dimension of index space (numeric or symbolic) */
        _idx(const ex & v, const ex & dim)

        bool info(unsigned inf) except +
        size_t nops() except +
        ex op(size_t i) except +
        ex map(imap_function & f) except +
        ex evalf() except +
        ex subs(const exmap & m, unsigned options) except +
#        /** Check whether the index forms a dummy index pair with another index
#	 *  of the same type. */
        bool is_dummy_pair_same_type(const basic & other) except +
#	// non-virtual functions in this class
#	/** Get value of index. */
        ex get_value() except +
#	/** Check whether the index is numeric. */
        bool is_numeric() except +
#	/** Check whether the index is symbolic. */
        bool is_symbolic() except +
#	/** Get dimension of index space. */
        ex get_dim() except +
#	/** Check whether the dimension is numeric. */
        bool is_dim_numeric() except +
#	/** Check whether the dimension is symbolic. */
        bool is_dim_symbolic() except +
#	/** Make a new index with the same value but a different dimension. */
        ex replace_dim(const ex & new_dim) except +
#	/** Return the minimum of the dimensions of this and another index.
#	 *  If this is undecidable, throw an exception. */
        ex minimal_dim(const _idx & other) except +

    cdef cppclass _varidx "ginacsym::varidx"(_idx):
        _varidx()
        _varidx(const ex & v, const ex & dim, bool covariant)
#	/** Check whether the index is covariant. */
        bool is_covariant() except +
#	/** Check whether the index is contravariant (not covariant) except +. */
        bool is_contravariant() except +
#	/** Make a new index with the same value but the opposite variance. */
        ex toggle_variance() except +

    cdef cppclass _spinidx "ginacsym::spinidx"(_varidx):
        _spinidx()
        _spinidx(const ex & v, const ex & dim, bool covariant, bool dotted)
        ex conjugate() except +
        bool is_dotted() except +
        bool is_undotted() except +
        ex toggle_dot() except +
        ex toggle_variance_dot() except +

#    // utility functions
#    /** Check whether two indices form a dummy pair. */
    bool _is_dummy_pair "is_dummy_pair"(const _idx & i1, const _idx & i2) except +
#    /** Check whether two expressions form a dummy index pair. */
    bool _is_dummy_pair "is_dummy_pair"(const ex & e1, const ex & e2) except +
#    /** Given a vector of indices, split them into two vectors, one containing
#     *  the free indices, the other containing the dummy indices (numeric
#     *  indices are neither free nor dummy ones).
#     *
#     *  @param it Pointer to start of index vector
#     *  @param itend Pointer to end of index vector
#     *  @param out_free Vector of free indices (returned, sorted)
#     *  @param out_dummy Vector of dummy indices (returned, sorted) */
#    void find_free_and_dummy(exvector::const_iterator it, exvector::const_iterator itend, exvector & out_free, exvector & out_dummy);

#    /** Given a vector of indices, split them into two vectors, one containing
#     *  the free indices, the other containing the dummy indices (numeric
#     *  indices are neither free nor dummy ones).
#     *
#     *  @param v Index vector
#     *  @param out_free Vector of free indices (returned, sorted)
#     *  @param out_dummy Vector of dummy indices (returned, sorted) */
    void _find_free_and_dummy "find_free_and_dummy"(const exvector & v, exvector & out_free, exvector & out_dummy) except +
#    /** Given a vector of indices, find the dummy indices.
#     *
#     *  @param v Index vector
#     *  @param out_dummy Vector of dummy indices (returned, sorted) */
    void _find_dummy_indices "find_dummy_indices"(const exvector & v, exvector & out_dummy) except +
#    /** Count the number of dummy index pairs in an index vector. */
    size_t _count_dummy_indices "count_dummy_indices"(const exvector & v) except +
#    /** Count the number of dummy index pairs in an index vector. */
    size_t _count_free_indices "count_free_indices"(const exvector & v) except +
#    /** Return the minimum of two index dimensions. If this is undecidable,
#     *  throw an exception. Numeric dimensions are always considered "smaller"
#     *  than symbolic dimensions. */
    ex _minimal_dim "minimal_dim"(const ex & dim1, const ex & dim2) except +

cdef extern from "ginacsym/indexed.h" namespace "ginacsym":
    cdef cppclass _indexed "ginacsym::indexed":
        _indexed()
#        /** Construct indexed object with no index.
#	 *
#	 *  @param b Base expression */
        _indexed(const ex & b) except +
#	/** Construct indexed object with one index. The index must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param i1 The index */
        _indexed(const ex & b, const _varidx & i1) except +
#	/** Construct indexed object with two indices. The indices must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param i1 First index
#	 *  @param i2 Second index */
        _indexed(const ex & b, const _varidx & i1, const _varidx & i2) except +
#	/** Construct indexed object with three indices. The indices must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param i1 First index
#	 *  @param i2 Second index
#	 *  @param i3 Third index */
        _indexed(const ex & b, const _varidx & i1, const _varidx & i2, const _varidx & i3) except +
#	/** Construct indexed object with four indices. The indices must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param i1 First index
#	 *  @param i2 Second index
#	 *  @param i3 Third index
#	 *  @param i4 Fourth index */
        _indexed(const ex & b, const _varidx & i1, const _varidx & i2, const _varidx & i3, const _varidx & i4) except +
#	/** Construct indexed object with two indices and a specified symmetry. The
#	 *  indices must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param symm Symmetry of indices
#	 *  @param i1 First index
#	 *  @param i2 Second index */
        _indexed(const ex & b, const _symmetry & symm, const _varidx & i1, const _varidx & i2) except +
#	/** Construct indexed object with three indices and a specified symmetry.
#	 *  The indices must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param symm Symmetry of indices
#	 *  @param i1 First index
#	 *  @param i2 Second index
#	 *  @param i3 Third index */
        _indexed(const ex & b, const _symmetry & symm, const _varidx & i1, const _varidx & i2, const _varidx & i3) except +
#	/** Construct indexed object with four indices and a specified symmetry. The
#	 *  indices must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param symm Symmetry of indices
#	 *  @param i1 First index
#	 *  @param i2 Second index
#	 *  @param i3 Third index
#	 *  @param i4 Fourth index */
        _indexed(const ex & b, const _symmetry & symm, const _varidx & i1, const _varidx & i2, const _varidx & i3, const _varidx & i4) except +
#	/** Construct indexed object with a specified vector of indices. The indices
#	 *  must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param iv Vector of indices */
        _indexed(const ex & b, const exvector & iv) except +
#	/** Construct indexed object with a specified vector of indices and
#	 *  symmetry. The indices must be of class idx.
#	 *
#	 *  @param b Base expression
#	 *  @param symm Symmetry of indices
#	 *  @param iv Vector of indices */
        _indexed(const ex & b, const _symmetry & symm, const exvector & iv) except +

        unsigned precedence() except +
        bool info(unsigned inf) except +
        ex eval() except +
        ex real_part() except +
        ex imag_part() except +
        exvector get_free_indices() except +

#        /** Check whether all index values have a certain property.
#	 *  @see class info_flags */
        bool all_index_values_are(unsigned inf) except +
#	/** Return a vector containing the object's indices. */
        exvector get_indices() except +
#	/** Return a vector containing the dummy indices of the object, if any. */
        exvector get_dummy_indices() except +
#	/** Return a vector containing the dummy indices in the contraction with
#	 *  another indexed object. This is symmetric: a.get_dummy_indices(b)
#	 *  == b.get_dummy_indices(a) */
        exvector get_dummy_indices(const _indexed & other) except +
#	/** Check whether the object has an index that forms a dummy index pair
#	 *  with a given index. */
        bool has_dummy_index_for(const ex & i) except +
#	/** Return symmetry properties. */
        ex get_symmetry() except +

cdef extern from "ginacsym/indexed.h" namespace "ginacsym":
    cdef cppclass _scalar_products "ginacsym::scalar_products":
        _scalar_products()
#        /** Register scalar product pair and its value. */
        void add(const ex & v1, const ex & v2, const ex & sp) except +

#	/** Register scalar product pair and its value for a specific space dimension. */
        void add(const ex & v1, const ex & v2, const ex & dim, const ex & sp) except +

#	/** Register list of vectors. This adds all possible pairs of products
#	 *  a.i * b.i with the value a*b (note that this is not a scalar vector
#	 *  product but an ordinary product of scalars). */
#	void add_vectors(const lst & l, const ex & dim);

#	/** Clear all registered scalar products. */
        void clear() except +

        bool is_defined(const ex & v1, const ex & v2, const ex & dim) except +
        ex evaluate(const ex & v1, const ex & v2, const ex & dim) except +

    #// utility functions

    #/** Returns all dummy indices from the expression */
    exvector _get_all_dummy_indices "get_all_dummy_indices"(const ex & e) except +

    #/** More reliable version of the form. The former assumes that e is an
    #  * expanded expression. */
    exvector _get_all_dummy_indices_safely "get_all_dummy_indices_safely"(const ex & e) except +

    #/** Returns b with all dummy indices, which are listed in va, renamed
    # *  if modify_va is set to TRUE all dummy indices of b will be appended to va */
    ex _rename_dummy_indices_uniquely "rename_dummy_indices_uniquely"(exvector & va, const ex & b, bool modify_va) except +

    #/** Returns b with all dummy indices, which are common with a, renamed */
    ex _rename_dummy_indices_uniquely "rename_dummy_indices_uniquely"(const ex & a, const ex & b) except +

    #/** Same as above, where va and vb contain the indices of a and b and are sorted */
    ex _rename_dummy_indices_uniquely "rename_dummy_indices_uniquely"(const exvector & va, const exvector & vb, const ex & b) except +

    #/** Similar to above, where va and vb are the same and the return value is a list of two lists
    # *  for substitution in b */
    #lst rename_dummy_indices_uniquely(const exvector & va, const exvector & vb)

    #/** This function returns the given expression with expanded sums
    # *  for all dummy index summations, where the dimensionality of
    # *  the dummy index is a nonnegative integer.
    # *  Optionally all indices with a variance will be substituted by
    # *  indices with the corresponding numeric values without variance.
    # *
    # *  @param e the given expression
    # *  @param subs_idx indicates if variance of dummy indices should be neglected
    # */
    ex _expand_dummy_sum "expand_dummy_sum"(const ex & e, bool subs_idx) except +

cdef extern from "ginacsym/tensor.h" namespace "ginacsym":
    cdef cppclass _tensor "ginacsym::tensor"(basic):
        pass
#    cdef cppclass tensdelta(tensor):
#        pass
    cdef cppclass tensmetric(_tensor):
        pass
    cdef cppclass _minkmetric "ginacsym::minkmetric"(tensmetric):
        _minkmetric()
        _minkmetric(bool pos_sig)
#    cdef cppclass spinmetric(tensmetric):
#        pass
#    cdef cppclass tensepsilon(tensor):
#        pass

    ## utility functions

#    /** Create a delta tensor with specified indices. The indices must be of class
#     *  idx or a subclass. The delta tensor is always symmetric and its trace is
#     *  the dimension of the index space.
#     *
#     *  @param i1 First index
#     *  @param i2 Second index
#     *  @return newly constructed delta tensor */
    ex _delta_tensor "ginacsym::delta_tensor"(const ex & i1, const ex & i2) except +

#    /** Create a symmetric metric tensor with specified indices. The indices
#     *  must be of class varidx or a subclass. A metric tensor with one
#     *  covariant and one contravariant index is equivalent to the delta tensor.
#     *
#     *  @param i1 First index
#     *  @param i2 Second index
#     *  @return newly constructed metric tensor */
    ex _metric_tensor "ginacsym::metric_tensor"(const ex & i1, const ex & i2) except +

#    /** Create a Minkowski metric tensor with specified indices. The indices
#     *  must be of class varidx or a subclass. The Lorentz metric is a symmetric
#     *  tensor with a matrix representation of diag(1,-1,-1,...) (negative
#     *  signature, the default) or diag(-1,1,1,...) (positive signature).
#     *
#     *  @param i1 First index
#     *  @param i2 Second index
#     *  @param pos_sig Whether the signature is positive
#     *  @return newly constructed Lorentz metric tensor */
    ex _lorentz_g "ginacsym::lorentz_g"(const ex & i1, const ex & i2, bool pos_sig) except +

#    /** Create a spinor metric tensor with specified indices. The indices must be
#     *  of class spinidx or a subclass and have a dimension of 2. The spinor
#     *  metric is an antisymmetric tensor with a matrix representation of
#     *  [[ [[ 0, 1 ]], [[ -1, 0 ]] ]].
#     *
#     *  @param i1 First index
#     *  @param i2 Second index
#     *  @return newly constructed spinor metric tensor */
    ex _spinor_metric "ginacsym::spinor_metric"(const ex & i1, const ex & i2) except +

#    /** Create an epsilon tensor in a Euclidean space with two indices. The
#     *  indices must be of class idx or a subclass, and have a dimension of 2.
#     *
#     *  @param i1 First index
#     *  @param i2 Second index
#     *  @return newly constructed epsilon tensor */
    ex _epsilon_tensor "ginacsym::epsilon_tensor"(const ex & i1, const ex & i2) except +

#    /** Create an epsilon tensor in a Euclidean space with three indices. The
#     *  indices must be of class idx or a subclass, and have a dimension of 3.
#     *
#     *  @param i1 First index
#     *  @param i2 Second index
#     *  @param i3 Third index
#     *  @return newly constructed epsilon tensor */
    ex _epsilon_tensor "ginacsym::epsilon_tensor"(const ex & i1, const ex & i2, const ex & i3) except +

#    /** Create an epsilon tensor in a Minkowski space with four indices. The
#     *  indices must be of class varidx or a subclass, and have a dimension of 4.
#     *
#     *  @param i1 First index
#     *  @param i2 Second index
#     *  @param i3 Third index
#     *  @param i4 Fourth index
#     *  @param pos_sig Whether the signature of the metric is positive
#     *  @return newly constructed epsilon tensor */
    ex _lorentz_eps "ginacsym::lorentz_eps"(const ex & i1, const ex & i2, const ex & i3, const ex & i4, bool pos_sig) except +

cdef extern from "ginacsym/clifford.h" namespace "ginacsym":
    cdef cppclass _clifford "ginacsym::clifford"(_indexed):
        _clifford()
        _clifford(const ex & b, unsigned char rl) except +
        _clifford(const ex & b, const ex & mu,  const ex & metr, unsigned char rl, int comm_sign) except +

#	// internal constructors
        _clifford(unsigned char rl, const ex & metr, int comm_sign, const exvector & v);
        _clifford(unsigned char rl, const ex & metr, int comm_sign, exvector && v);

        unsigned char get_representation_label() except +
        ex get_metric() except +
        ex get_metric(const ex & i, const ex & j, bool symmetrised) except +
        bool same_metric(const ex & other) except +
        int get_commutator_sign() except +

#        // global functions

#        /** Check whether a given return_type_t object (as returned by return_type_tinfo()
#          * is that of a clifford object (with an arbitrary representation label).
#          *
#          * @param ti tinfo key */
    bool _is_clifford_tinfo "ginacsym::is_clifford_tinfo"(const _return_type_t& ti) except +


#        /** Create a Clifford unity object.
#         *
#         *  @param rl Representation label
#         *  @return newly constructed object */
    ex _dirac_ONE "ginacsym::dirac_ONE"(unsigned char rl)  except +

#        /** Create a Clifford unit object.
#         *
#         *  @param mu Index (must be of class varidx or a derived class)
#         *  @param metr Metric (should be indexed, tensmetric or a derived class, or a matrix)
#         *  @param rl Representation label
#         *  @return newly constructed Clifford unit object */
    ex _clifford_unit "ginacsym::clifford_unit"(const ex & mu, const ex & metr, unsigned char rl) except +

#        /** Create a Dirac gamma object.
#         *
#         *  @param mu Index (must be of class varidx or a derived class)
#         *  @param rl Representation label
#         *  @return newly constructed gamma object */
    ex _dirac_gamma "ginacsym::dirac_gamma"(const ex & mu, unsigned char rl) except +

#        /** Create a Dirac gamma5 object.
#         *
#         *  @param rl Representation label
#         *  @return newly constructed object */
    ex _dirac_gamma5 "ginacsym::dirac_gamma5"(unsigned char rl) except +

#        /** Create a Dirac gammaL object.
#         *
#         *  @param rl Representation label
#         *  @return newly constructed object */
    ex _dirac_gammaL "ginacsym::dirac_gammaL"(unsigned char rl) except +

#        /** Create a Dirac gammaR object.
#         *
#         *  @param rl Representation label
#         *  @return newly constructed object */
    ex _dirac_gammaR "ginacsym::dirac_gammaR"(unsigned char rl) except +

#        /** Create a term of the form e_mu * gamma~mu with a unique index mu.
#         *
#         *  @param e Original expression
#         *  @param dim Dimension of index
#         *  @param rl Representation label */
    ex _dirac_slash "ginacsym::dirac_slash"(const ex & e, const ex & dim, unsigned char rl) except +

#        /** Calculate dirac traces over the specified set of representation labels.
#         *  The computed trace is a linear functional that is equal to the usual
#         *  trace only in D = 4 dimensions. In particular, the functional is not
#         *  always cyclic in D != 4 dimensions when gamma5 is involved.
#         *
#         *  @param e Expression to take the trace of
#         *  @param rls Set of representation labels
#         *  @param trONE Expression to be returned as the trace of the unit matrix */
    ex _dirac_trace "ginacsym::dirac_trace"(const ex & e, const cppset[unsigned char] & rls, const ex & trONE) except +

#        /** Calculate dirac traces over the specified list of representation labels.
#         *  The computed trace is a linear functional that is equal to the usual
#         *  trace only in D = 4 dimensions. In particular, the functional is not
#         *  always cyclic in D != 4 dimensions when gamma5 is involved.
#         *
#         *  @param e Expression to take the trace of
#         *  @param rll List of representation labels
#         *  @param trONE Expression to be returned as the trace of the unit matrix */
    ex _dirac_trace "ginacsym::dirac_trace"(const ex & e, const _lst & rll, const ex & trONE) except +

#        /** Calculate the trace of an expression containing gamma objects with
#         *  a specified representation label. The computed trace is a linear
#         *  functional that is equal to the usual trace only in D = 4 dimensions.
#         *  In particular, the functional is not always cyclic in D != 4 dimensions
#         *  when gamma5 is involved.
#         *
#         *  @param e Expression to take the trace of
#         *  @param rl Representation label
#         *  @param trONE Expression to be returned as the trace of the unit matrix */
    ex _dirac_trace "ginacsym::dirac_trace"(const ex & e, unsigned char rl, const ex & trONE) except +

#        /** Bring all products of clifford objects in an expression into a canonical
#         *  order. This is not necessarily the most simple form but it will allow
#         *  to check two expressions for equality. */
    ex _canonicalize_clifford "ginacsym::canonicalize_clifford"(const ex & e) except +

#        /** Automorphism of the Clifford algebra, simply changes signs of all
#         *  clifford units. */
    ex _clifford_prime "ginacsym::clifford_prime"(const ex & e) except +

#        /** An auxillary function performing clifford_star() and clifford_bar().*/
    ex _clifford_star_bar "ginacsym::clifford_star_bar"(const ex & e, bool do_bar, unsigned options) except +

#        /** Main anti-automorphism of the Clifford algebra: makes reversion
#         *  and changes signs of all clifford units. */
    ex _clifford_bar "ginacsym::clifford_bar"(const ex & e) except +

#        /** Reversion of the Clifford algebra, reverse the order of all clifford units
#         *  in ncmul. */
    ex _clifford_star "ginacsym::clifford_star"(const ex & e) except +

#        /** Replaces dirac_ONE's (with a representation_label no less than rl) in e with 1.
#         *  For the default value rl = 0 remove all of them. Aborts if e contains any
#         *  clifford_unit with representation_label to be removed.
#         *
#         *  @param e Expression to be processed
#         *  @param rl Value of representation label
#         *  @param options Defines some internal use */
    ex _remove_dirac_ONE "ginacsym::remove_dirac_ONE"(const ex & e, unsigned char rl, unsigned options) except +

#        /** Returns the maximal representation label of a clifford object
#         *  if e contains at least one, otherwise returns -1
#         *
#         *  @param e Expression to be processed
#         *  @param ignore_ONE defines if clifford_ONE should be ignored in the search */
    int _clifford_max_label "ginacsym::clifford_max_label"(const ex & e, bool ignore_ONE) except +

#        /** Calculation of the norm in the Clifford algebra. */
    ex _clifford_norm "ginacsym::clifford_norm"(const ex & e) except +

#        /** Calculation of the inverse in the Clifford algebra. */
    ex _clifford_inverse "ginacsym::clifford_inverse"(const ex & e) except +

#        /** List or vector conversion into the Clifford vector.
#         *
#         *  @param v List or vector of coordinates
#         *  @param mu Index (must be of class varidx or a derived class)
#         *  @param metr Metric (should be indexed, tensmetric or a derived class, or a matrix)
#         *  @param rl Representation label
#         *  @return Clifford vector with given components */
    ex _lst_to_clifford "ginacsym::lst_to_clifford"(const ex & v, const ex & mu,  const ex & metr, unsigned char rl) except +

#        /** List or vector conversion into the Clifford vector.
#         *
#         *  @param v List or vector of coordinates
#         *  @param e Clifford unit object
#         *  @return Clifford vector with given components */
    ex _lst_to_clifford "ginacsym::lst_to_clifford"(const ex & v, const ex & e) except +

#        /** An inverse function to lst_to_clifford(). For given Clifford vector extracts
#         *  its components with respect to given Clifford unit. Obtained components may
#         *  contain Clifford units with a different metric. Extraction is based on
#         *  the algebraic formula (e * c.i + c.i * e)/ pow(e.i, 2) for non-degenerate cases
#         *  (i.e. neither pow(e.i, 2) = 0).
#         *
#         *  @param e Clifford expression to be decomposed into components
#         *  @param c Clifford unit defining the metric for splitting (should have numeric dimension of indices)
#         *  @param algebraic Use algebraic or symbolic algorithm for extractions
#         *  @return List of components of a Clifford vector*/
    _lst _clifford_to_lst "ginacsym::clifford_to_lst"(const ex & e, const ex & c, bool algebraic) except +

#        /** Calculations of Moebius transformations (conformal map) defined by a 2x2 Clifford matrix
#         *  (a b\\c d) in linear spaces with arbitrary signature. The expression is
#         *  (a * x + b)/(c * x + d), where x is a vector build from list v with metric G.
#         *  (see Jan Cnops. An introduction to {D}irac operators on manifolds, v.24 of
#         *  Progress in Mathematical Physics. Birkhauser Boston Inc., Boston, MA, 2002.)
#         *
#         *  @param a (1,1) entry of the defining matrix
#         *  @param b (1,2) entry of the defining matrix
#         *  @param c (2,1) entry of the defining matrix
#         *  @param d (2,2) entry of the defining matrix
#         *  @param v Vector to be transformed
#         *  @param G Metric of the surrounding space, may be a Clifford unit then the next parameter is ignored
#         *  @param rl Representation label
#         *  @return List of components of the transformed vector*/
    ex _clifford_moebius_map "ginacsym::clifford_moebius_map"(const ex & a, const ex & b, const ex & c, const ex & d, const ex & v, const ex & G, unsigned char rl) except +

#        /** The second form of Moebius transformations defined by a 2x2 Clifford matrix M
#         *  This function takes the transformation matrix M as a single entity.
#         *
#         *  @param M the defining matrix
#         *  @param v Vector to be transformed
#         *  @param G Metric of the surrounding space, may be a Clifford unit then the next parameter is ignored
#         *  @param rl Representation label
#         *  @return List of components of the transformed vector*/
    ex _clifford_moebius_map "ginacsym::clifford_moebius_map"(const ex & M, const ex & v, const ex & G, unsigned char rl) except +

cdef extern from "ginacsym/color.h" namespace "ginacsym":
    cdef cppclass _color "ginacsym::color"(_indexed):
        _color()
        _color(const ex & b, unsigned char rl) except +
        _color(const ex & b, const ex & i1, unsigned char rl) except +
        unsigned char get_representation_label() except +


    #// global functions

    #/** Create the su(3) unity element. This is an indexed object, although it
    # *  has no indices.
    # *
    # *  @param rl Representation label
    # *  @return newly constructed unity element */
    ex _color_ONE "ginacsym::color_ONE"(unsigned char rl) except +

    #/** Create an su(3) generator.
    # *
    # *  @param a Index
    # *  @param rl Representation label
    # *  @return newly constructed unity generator */
    ex _color_T "ginacsym::color_T"(const ex & a, unsigned char rl) except +

    #/** Create an su(3) antisymmetric structure constant.
    # *
    # *  @param a First index
    # *  @param b Second index
    # *  @param c Third index
    # *  @return newly constructed structure constant */
    ex _color_f "ginacsym::color_f"(const ex & a, const ex & b, const ex & c) except +

    #/** Create an su(3) symmetric structure constant.
    # *
    # *  @param a First index
    # *  @param b Second index
    # *  @param c Third index
    # *  @return newly constructed structure constant */
    ex _color_d "ginacsym::color_d"(const ex & a, const ex & b, const ex & c) except +

    #/** This returns the linear combination d.a.b.c+I*f.a.b.c. */
    ex _color_h "ginacsym::color_h"(const ex & a, const ex & b, const ex & c) except +

    #/** Calculate color traces over the specified set of representation labels.
    # *
    # *  @param e Expression to take the trace of
    # *  @param rls Set of representation labels */
    ex _color_trace "ginacsym::color_trace"(const ex & e, const cppset[unsigned char] & rls) except +

    #/** Calculate color traces over the specified list of representation labels.
    # *
    # *  @param e Expression to take the trace of
    # *  @param rll List of representation labels */
    ex _color_trace "ginacsym::color_trace"(const ex & e, const _lst & rll) except +

    #/** Calculate the trace of an expression containing color objects with a
    # *  specified representation label.
    # *
    # *  @param e Expression to take the trace of
    # *  @param rl Representation label */
    ex _color_trace "ginacsym::color_trace"(const ex & e, unsigned char rl) except +

cdef extern from "ginacsym/fail.h" namespace "ginacsym":
    cdef cppclass _fail "ginacsym::fail"(basic):
        _fail()
#    cdef ex _fail_to_ex(const _fail& f)

cdef extern from "ginacsym/inifcns.h" namespace "ginacsym":
    ex _sqrt "sqrt"(const ex& e) except +
    ex _abs "abs"(const ex& e) except +
    ex _conjugate "conjugate"(const ex& e) except +
    ex _real_part "real_part"(const ex& e) except +
    ex _imag_part "imag_part"(const ex& e) except +
    ex _step "step"(const ex& e) except +
    ex _csgn "csgn"(const ex& e) except +
    ex _eta "eta"(const ex& e1,const ex& e2) except +

    ex _exp "exp"(const ex& e) except +
    ex _log "log"(const ex& e) except + #natural polylogarithm
    ex _logb "logb"(const ex& e,const ex& b) except +

    ex _sin "sin"(const ex& e) except +
    ex _cos "cos"(const ex& e) except +
    ex _tan "tan"(const ex& e) except +
    ex _sec "sec"(const ex& e) except +
    ex _csc "csc"(const ex& e) except +
    ex _cot "cot"(const ex& e) except +

    ex _arcsin "asin"(const ex& e) except +
    ex _arccos "acos"(const ex& e) except +
    ex _arctan "atan"(const ex& e) except +
    ex _arcsec "asec"(const ex& e) except +
    ex _arccsc "acsc"(const ex& e) except +
    ex _arccot "acot"(const ex& e) except +

    ex _sinh "sinh"(const ex& e) except +
    ex _cosh "cosh"(const ex& e) except +
    ex _tanh "tanh"(const ex& e) except +
    ex _sech "sech"(const ex& e) except +
    ex _csch "csch"(const ex& e) except +
    ex _coth "coth"(const ex& e) except +

    ex _arcsinh "asinh"(const ex& e) except +
    ex _arccosh "acosh"(const ex& e) except +
    ex _arctanh "atanh"(const ex& e) except +
    ex _arcsech "asech"(const ex& e) except +
    ex _arccsch "acsch"(const ex& e) except +
    ex _arccoth "acoth"(const ex& e) except +

    ex _Li "Li"(const ex& m,const ex& x) except + #classical polylogarithm as well as multiple polylogarithm
    ex _Li2 "Li2"(const ex& e) except +
    ex _Li3 "Li3"(const ex& e) except +
    ex _zetaderiv "zetaderiv"(const ex& n,const ex& x) except + # Derivatives of Riemann's Zeta-function  zetaderiv(0,x)==zeta(x)
    ex _zeta "zeta"(const ex& m) except +
    ex _zeta "zeta"(const ex& m,const ex& s) except +
    ex _G "G"(const ex& a,const ex& y) except + #multiple polylogarithm
    ex _G "G"(const ex& a,const ex& s,const ex& y) except + #multiple polylogarithm
    ex _S "S"(const ex& n,const ex& p,const ex& x) except + #Nielsen’s generalized polylogarithm
    ex _H "H"(const ex& m,const ex& x) except + #harmonic polylogarithm
    ex _tgamma "tgamma"(const ex& e) except +
    ex _lgamma "lgamma"(const ex& e) except +
    ex _beta "beta"(const ex& m,const ex& n) except +
    ex _psi "psi"(const ex& e) except + #(digamma) function
    ex _psi "psi"(const ex& n,const ex& x) except + #derivatives of psi function (polygamma functions)
    ex _factorial "factorial"(const ex& e) except +
    ex _binomial "binomial"(const ex& n,const ex& k) except + #binomial coefficients
    ex _Order "Order"(const ex& e) except + #order term function in truncated power series

    ex _EllipticK "EllipticK"(const ex& e) except + #Complete elliptic integral of the first kind.
    ex _EllipticE "EllipticE"(const ex& e) except + #Complete elliptic integral of the second kind.

    ex _chebyshevT "chebyshevT"(const ex& n,const ex& x) except +
    ex _chebyshevU "chebyshevU"(const ex& n,const ex& x) except +
    ex _legendreP "legendreP"(const ex& n,const ex& x) except +
    ex _hermiteH "hermiteH"(const ex& n,const ex& x) except +
    ex _gegenbauerC "gegenbauerC"(const ex& n,const ex& a,const ex& x) except +

    ex _sinIntegral "sinIntegral"(const ex& e) except +
    ex _cosIntegral "cosIntegral"(const ex& e) except +
    ex _sinhIntegral "sinhIntegral"(const ex& e) except +
    ex _coshIntegral "coshIntegral"(const ex& e) except +
    ex _logIntegral "logIntegral"(const ex& e) except +
    ex _expIntegralEi "expIntegralEi"(const ex& e) except +
    ex _expIntegralE "expIntegralE"(const ex& s, const ex& e) except +

    ex _legendreP "legendreP"(const ex& n,const ex& m,const ex& x) except +
    ex _legendreQ "legendreQ"(const ex& n,const ex& x) except +
    ex _legendreQ "legendreQ"(const ex& n,const ex& m,const ex& x) except +
    ex _besselJ "besselJ"(const ex& n,const ex& x) except +
    ex _besselY "besselY"(const ex& n,const ex& x) except +
    ex _besselI "besselI"(const ex& n,const ex& x) except +
    ex _besselK "besselK"(const ex& n,const ex& x) except +

#    ex _ ""(const ex& n,const ex& x) except +
#    ex _ ""(const ex& e) except +
#    ex _ ""(const ex& e) except +

    ex _lsolve "ginacsym::lsolve"(const ex &eqns, const ex &symbols, unsigned options) except +
    const _numeric _fsolve "ginacsym::fsolve"(const ex& f, const _symbol& x, const _numeric& x1, const _numeric& x2) except +
#    /** Converts a given list containing parameters for H in Remiddi/Vermaseren notation into
#     *  the corresponding ginacsym functions.
#     */
    ex  _convert_H_to_Li "ginacsym::convert_H_to_Li"(const ex& parameterlst, const ex& arg) except +

cdef extern from "ginacsym/function.h" namespace "ginacsym":
    cdef cppclass _function "ginacsym::function":
        string get_name() const

cdef extern from "ginacsym/ginacwrapper.h" namespace "ginacsym":
    cdef string to_string(const ex& expr) except +
    cdef string to_latex_string(const ex& expr) except +
    cdef string to_python_string(const ex& expr) except +
    cdef ex _matrix_to_ex(const _matrix& m) except +
    cdef ex _relational_to_ex(const _relational& r) except +
    cdef ex _pseries_to_ex(const _pseries& r) except +
    cdef ex _clifford_to_ex(const _clifford& r) except +
    cdef ex _minkmetric_to_ex(const _minkmetric& r) except +
    cdef _indexed _clifford_to__indexed(const _clifford& r) except +
    cdef ex _color_to_ex(const _color& r) except +
    cdef ex _ncmul_to_ex(const _ncmul& r) except +
    cdef _indexed _color_to__indexed(const _color& r) except +
#    bool _find(const ex& expr,const ex& pattern, vector[ex]& v) except +

    cdef bool is_lst(const ex& e) except+
    void prepend(ex & a,const ex & b) except +
    void append(ex & a,const ex & b) except +
    void remove_first(ex & a) except +
    void remove_last(ex & a) except +
    void remove_all(ex & a) except +
    void sort(ex & a) except +
    void unique(ex & a) except +

    cdef cppclass generatorc:
        generatorc()
        ex& symGenerator(const string& s, unsigned symboltype,const bool& islatexname) except +
        ex exGenerator(const string& s, unsigned symboltype, const bool& islatexname) except +
        ex functionSymbolFromString(const string& s, unsigned symboltype, const bool& islatexname)except +
        int symRegister(const ex& syms) except +
        ex exGeneratorFromString(const string& s) except +
        cppmap[ex,unsigned,ex_is_less] allinfo() except +
        cppmap[ex, unsigned, ex_is_less] aninfo(const ex& e) except +
    cdef generatorc generator
    cdef _matrix _Matrix(const cpplist[cpplist[ex]]& inpma) except +

cdef extern from "ginacsym/utility.h" namespace "ginacsym":
    ctypedef cppset[_lst] exsetlst
    exset get_symbols(const ex& e) except +
    ex collectAll(const ex& e,const ex& _var,bool _distributed) except +
    bool is_number(const ex& e) except +
    cdef void _set_digits(const long& prec) except +
    cdef long _get_digits() except +
    cdef ex f_map(const ex &e1,const ex &e2) except +
    cdef ex Gcd(_lst exp) except +
    cdef ex Lcm(_lst exp) except +

cdef extern from "ginacsym/infinity.h" namespace "ginacsym":
    cdef cppclass _infinity "ginacsym::infinity"(basic):
        ex eval()
    const _infinity _Infinity "ginacsym::Infinity"

cdef extern from "ginacsym/functions.h" namespace "ginacsym":
    cdef cppclass _functions "ginacsym::functions"(basic):
        _functions()
        _functions(const string &fns, const _lst &fd, unsigned assu) except +
        ex to_ex() except +
        ex total_diff(const ex & e) except +

cdef extern from "ginacsym/inert.h" namespace "ginacsym":
    cdef cppclass _Limit "ginacsym::Limit"(basic):
        _Limit()
        _Limit(const ex& e, const ex& z, const ex& z0, const string& dir) except +
        ex to_ex() except +

    cdef cppclass _Diff "ginacsym::Diff"(basic):
        _Diff()
        _Diff(const ex& d, const ex& i, const ex& o) except +
        ex changeVariable(const ex& oldNew, const ex& newvarName) except +
        ex to_ex() except +

    cdef cppclass _Integrate "ginacsym::Integrate"(basic):
        _Integrate()
        _Integrate(const ex& integrand_, const ex& var_,const int& partial_num) except +
        _Integrate(const ex& integrand_, const ex& var_,const ex& l_, const ex& u_,const int& partial_num) except +
        ex to_ex() except +
        void set_partial_num(const int& p) except +
        ex integrate(const ex& var_) except +
        ex integrate(const ex& var_,const ex& l_, const ex& u_) except +
        _Integrate changeVariable(const ex& oldNew, const ex& newvarName) except +
        ex evaluate() except +

    ex _evaluate "ginacsym::evaluate"(const ex& e) except +
    ex _apply_partial_integration_on_ex "ginacsym::apply_partial_integration_on_ex"(const ex& e, const unsigned& partial_num) except +

cdef extern from "ginacsym/integrate.h" namespace "ginacsym":
    ex _integrate "ginacsym::integrate"(const ex& expr_, const ex& var_, const int& partial_num) except +
    ex _integrate "ginacsym::integrate"(const ex& expr_, const ex& var_,const ex& l_, const ex& u_, const int& partial_num) except +

cdef extern from "ginacsym/limit.h" namespace "ginacsym":
    ex _limit "ginacsym::limit"(const ex& e, const ex& z, const ex& z0, const string& dir) except +

cdef extern from "ginacsym/simplify.h" namespace "ginacsym":
    cdef enum _simplify_options:
        _algSimp "ginacsym::simplify_options::AlgSimp"
        _trigSimp "ginacsym::simplify_options::TrigSimp"
        _trigCombine "ginacsym::simplify_options::TrigCombine"
        _logSimp "ginacsym::simplify_options::logSimp"
        _jacobiSimp "ginacsym::simplify_options::JacobiSimp"
        _algSimp2 "ginacsym::simplify_options::AlgSimp2"
        _hyperSimp "ginacsym::simplify_options::HyperSimp"
        _funcSimp "ginacsym::simplify_options::FuncSimp"
        _hyperCombine "ginacsym::simplify_options::HyperCombine"

    ex Collect_common_factor(const ex& e) except +
    ex _fullSimplify "ginacsym::fullSimplify"(const ex& expr_, unsigned int rules) except +
    ex _simplify "ginacsym::simplify"(const ex& expr_, unsigned int rules) except +
    ex _fullsimplify "ginacsym::fullsimplify"(const ex& expr_, unsigned int rules) except +

cdef extern from "ginacsym/solve.h" namespace "ginacsym":
    exsetlst _solve "ginacsym::solve"(const _lst& equs_, const _lst& vars_) except +

##################### pyx  ##############################################

############ assumptions ###################################

cdef dict ex_map_to_Ex_map(cppmap[ex,unsigned,ex_is_less] inp):
    cdef cppmap[ex,unsigned,ex_is_less].iterator itr=inp.begin()
    cdef dict clt={}
    while itr!=inp.end():
        if deref(itr).second==_symbol_assumptions._symbol_assumptions_symbol:
            clt[ex_to_Ex(deref(itr).first)]="complex"
        elif deref(itr).second==_symbol_assumptions._realsymbol:
            clt[ex_to_Ex(deref(itr).first)]="real"
        elif deref(itr).second==_symbol_assumptions._possymbol:
            clt[ex_to_Ex(deref(itr).first)]="positive"
        inc(itr)
    return clt

cdef dict get_assumptions(ex e, bool isallinfo):
    cdef cppmap[ex,unsigned,ex_is_less] syminfomap
    if isallinfo:
        syminfomap = generator.allinfo()
    else:
        syminfomap = generator.aninfo(e)
    return ex_map_to_Ex_map(syminfomap)
cpdef dict all_symbols():
    return get_assumptions(ex(0), True)

###### enum options ###################
cpdef enum operators:
    equal
    not_equal
    less
    less_or_equal
    greater
    greater_or_equal

cpdef enum gcd_options:
    gcd_nooptions = 0
    gcd_options_no_heur_gcd = _no_heur_gcd
    gcd_options_no_part_factored = _no_part_factored
    gcd_options_use_sr_gcd = _use_sr_gcd

cpdef enum symbol_assumptions:
    complex = _symbol_assumptions_symbol
    real = _realsymbol
    positive = _possymbol

cpdef enum expand_options:
    expand_nooptions = 0
    expand_options_expand_indexed = _expand_indexed
    expand_options_expand_function_args = _expand_function_args
#    expand_options_expand_rename_idx = _expand_rename_idx
    expand_options_expand_transcendental = _expand_transcendental

cpdef enum has_options:
    has_nooptions = 0
    has_options_algebraic = _has_algebraic

cpdef enum subs_options:
    subs_nooptions = 0
    subs_options_no_pattern = _no_pattern
    subs_options_algebraic = _algebraic
    subs_options_no_index_renaming = _no_index_renaming
    subs_options_really_subs_idx = _really_subs_idx

cpdef enum series_options:
    series_nooptions = 0
    series_options_suppress_branchcut = _suppress_branchcut

cpdef enum factor_options:
    factor_nooptions = 0
    factor_options_polynomial = _polynomial
    factor_options_all = _all

cpdef enum determinant_algo:
    determinant_algo_automatic= _determinant_algo_automatic
    determinant_algo_gauss = _determinant_algo_gauss
    determinant_algo_divfree = _determinant_algo_divfree
    determinant_algo_laplace = _determinant_algo_laplace
    determinant_algo_bareiss = _determinant_algo_bareiss

cpdef enum solve_algo:
    solve_algo_automatic = _solve_algo_automatic
    solve_algo_gauss = _solve_algo_gauss
    solve_algo_divfree = _solve_algo_divfree
    solve_algo_bareiss = _solve_algo_bareiss
    solve_algo_markowitz = _solve_algo_markowitz

cpdef enum symmetry_type:
    symmetry_type_none         #/**< no symmetry properties */
    symmetry_type_symmetric     #/**< totally symmetric */
    symmetry_type_antisymmetric #/**< totally antisymmetric */
    symmetry_type_cyclic         #/**< cyclic symmetry */

cpdef enum info_flags:
    info_flags_numeric = _info_flags_numeric
    info_flags_real = _real
    info_flags_rational = _rational
    info_flags_integer = _integer
    info_flags_crational = _crational
    info_flags_cinteger = _cinteger
    info_flags_positive = _positive
    info_flags_negative = _negative
    info_flags_nonnegative = _nonnegative
    info_flags_posint = _posint
    info_flags_negint = _negint
    info_flags_nonnegint = _nonnegint
    info_flags_even = _even
    info_flags_odd = _odd
    info_flags_prime = _prime

    info_flags_relation = _relation
    info_flags_relation_equal = _relation_equal
    info_flags_relation_not_equal = _relation_not_equal
    info_flags_relation_less = _relation_less
    info_flags_relation_less_or_equal = _relation_less_or_equal
    info_flags_relation_greater = _relation_greater
    info_flags_relation_greater_or_equal = _relation_greater_or_equal

    info_flags_symbol = _info_flags_symbol
    info_flags_lst = _list
#    info_flags_exprseq = _info_flags_exprseq

    info_flags_polynomial = _info_flags_polynomial
    info_flags_integer_polynomial = _integer_polynomial
    info_flags_cinteger_polynomial = _cinteger_polynomial
    info_flags_rational_polynomial = _rational_polynomial
    info_flags_crational_polynomial = _crational_polynomial
    info_flags_rational_function = _rational_function

#    info_flags_indexed = _info_flags_indexed
#    info_flags_has_indices = _has_indices
#    info_flags_idx = _info_flags_idx
#    info_flags_expanded = _expanded
#    info_flags_indefinite = _indefinite

cpdef enum return_types:
    return_types_commutative = _commutative
    return_types_noncommutative = _noncommutative
    return_types_noncommutative_composite = _noncommutative_composite

cpdef enum simplify_options:
    simplify_nooptions = 0
    simplify_options_trigCombine = _trigCombine
    simplify_options_hyperCombine = _hyperCombine

cpdef enum Ex_iterator:
    const_iterator
    const_preorder_iterator
    const_postorder_iterator

#### cpp containers ###########


####### Ex_to_ex and py_to_ex ########
cdef ex py_to_ex(inp):
    if isinstance(inp,int) or isinstance(inp,float) or isinstance(inp,str):
        return (Ex(inp)._this)
    elif isinstance(inp,Ex):
        return copy_Ex(inp)._this
    else:
        raise Exception("unsupported variable.")

cdef cppset[unsigned char] set_int_to_cppset(set inp):
    cdef cppset[unsigned char] clt
    for i in inp:
        clt.insert(int(i))
    return clt

cdef cpplist[ex] list_to_cpplist(list inp):
    cdef cpplist[ex] clt
    for i in inp:
        if isinstance(i,int) or isinstance(i,float) or isinstance(i,str):
            clt.push_back(Ex(i)._this)
        else:
            clt.push_back(copy_Ex(i)._this)
    return clt

cdef exvector list_to_exvector(list inp):
    cdef exvector clt
    for i in inp:
        if isinstance(i,int) or isinstance(i,float) or isinstance(i,str):
            clt.push_back(Ex(i)._this)
        else:
            clt.push_back(copy_Ex(i)._this)
    return clt

cdef exvector varidx_list_to_exvector(list inp):
    cdef exvector clt
    for i in inp:
        clt.push_back((copy_varidx(i)._thisVaridx).evalf())
    return clt

cdef exvector spinidx_list_to_exvector(list inp):
    cdef exvector clt
    for i in inp:
        clt.push_back(copy_spinidx(i)._thisSpinidx.evalf())
    return clt

cdef exvector idx_list_to_exvector(list inp):
    cdef exvector clt
    for i in inp:
        clt.push_back((copy_idx(i)._thisIdx).evalf())
    return clt

cdef exmap dict_to_exmap(dict inp):
    cdef exmap clt
    for i in inp.iterkeys():
        if isinstance(inp[i],int) or isinstance(inp[i],float) or isinstance(inp[i],str):
            clt[copy_Ex(i)._this]=Ex(inp[i])._this
        else:
            clt[copy_Ex(i)._this]=copy_Ex(inp[i])._this
    return clt

cdef epvector list_to_epvector(list inp):
    cdef epvector vecclt
    for i in inp:
        vecclt.push_back(expair(py_to_ex(i[0]),py_to_ex(i[1])))
    return vecclt

cdef _lst list_to__lst(list inp):
    cdef cpplist[ex] tem= list_to_cpplist(inp)
    return cpplist_to__lst(tem)

######## ex_to_Ex ##########

cdef list cpplist_ex_to_Ex_list(cpplist[ex] inp):
    cdef list clt=[]
    cdef cpplist[ex].iterator itr=inp.begin()
    while itr!=inp.end():
        clt.append(ex_to_Ex(deref(itr)))
        inc(itr)
    return clt

cdef list exvector_to_Ex_list(vector[ex] inp):
    cdef list clt=[]
    cdef vector[ex].iterator itr=inp.begin()
    while itr!=inp.end():
        clt.append(ex_to_Ex(deref(itr)))
        inc(itr)
    return clt

cdef list _lst_to_list(_lst inp):
    cdef list clt=[]
    cdef ex inpex=_lst_to_ex(inp)
    itr=inpex.begin()
    while itr != inpex.end():
        clt.append(ex_to_Ex(deref(itr)))
        inc(itr)
    return clt

cdef lst exsetlst_to_lst(exsetlst inp):
    cdef lst clt=lst([])
    itr=inp.begin()
    while itr!=inp.end():
        clt.append(ex_to_Ex(_lst_to_ex(deref(itr))))
        inc(itr)
    return clt

cdef lst exset_to_lst(exset inp):
    cdef lst clt=lst([])
    itr=inp.begin()
    while itr!=inp.end():
        clt.append(ex_to_Ex(deref(itr)))
        inc(itr)
    return clt
cdef lst _lst_to_lst(_lst inp):
    cdef lst ret=lst(None)
    ret._this=_lst_to_ex(inp)
    return ret

cdef pseries _pseries_to_pseries(_pseries inp):
    cdef pseries ret=pseries(None)
    ret._thisPseries = inp
    ret._this = _pseries_to_ex(inp)
    return ret

cdef Ex ex_to_Ex(ex other):
    cdef Ex ret=Ex(None)
    ret._this=other
    return ret

cdef lst ex_to_lst(ex other):
    cdef lst ret=lst(None)
    ret._this=other
    ret._thisLst=ex_to[_lst](other)
    return ret

cdef numeric _numeric_to_numeric(_numeric other):
    cdef numeric ret=numeric(None)
    ret._thisNumeric=other
    ret._this=other.eval()
    return ret
cdef relational _relational_to_relational(_relational other):
    cdef relational ret=relational(None)
    ret._thisRelational=other
    ret._this=_relational_to_ex(other)
    return ret
cdef matrix _matrix_to_matrix(_matrix other):
    cdef matrix ret=matrix(None)
    ret._thisMatrix=other
    ret._this=_matrix_to_ex(other)
    return ret
cdef idx _idx_to_idx(_idx other):
    cdef idx ret=idx(None)
    ret._thisIdx=other
    ret._this=other.evalf()
    return ret
cdef varidx _varidx_to_varidx(_varidx other):
    cdef varidx ret=varidx(None)
    ret._thisVaridx=other
    ret._this=other.evalf()
    ret._thisIdx=ex_to[_idx](ret._this)
    return ret
cdef spinidx _spinidx_to_spinidx(_spinidx other):
    cdef spinidx ret=spinidx(None)
    ret._thisSpinidx=other
    ret._this=other.evalf()
    ret._thisVaridx=ex_to[_varidx](ret._this)
    ret._thisIdx=ex_to[_idx](ret._this)
    return ret
cdef indexed _indexed_to_indexed(_indexed other):
    cdef indexed ret=indexed(None)
    ret._thisIndexed=other
    ret._this=other.eval()
    return ret
cdef functions _functions_to_functions(_functions other):
    cdef functions ret=functions(None)
    ret._thisFunctions=other
    ret._this=other.to_ex()
    return ret
cdef Diff _Diff_to_Diff(_Diff other):
    cdef Diff ret=Diff(None)
    ret._thisDiff=other
    ret._this=other.to_ex()
    return ret
cdef Integrate _Integrate_to_Integrate(_Integrate other):
    cdef Integrate ret=Integrate(None)
    ret._thisIntegrate = other
    ret._this = other.to_ex()
    return ret

cdef return_type_t _return_type_t_to_return_type_t(_return_type_t other):
    cdef return_type_t ret=return_type_t(None)
    ret._thisReturn_type_t=other
    return ret


####Ex_to_Ex #########
cpdef dict lst_to_dict(lst inp):
    cdef dict tem={}
    for i in range(inp.nops()):
        tem[inp.op(i).lhs()]=inp.op(i).rhs()
    return tem

cdef list Ex_lst_to_Ex_list(Ex inp):
    cdef list tem=[]
    for i in range(inp.nops()):
        tem.append(inp.op(i))
    return tem

cdef Ex copy_Ex(Ex inp):
    if inp is None:
        raise Exception("unable to copy Ex, because it is None.")
    return inp

cdef lst copy_lst(lst inp):
    return inp

cdef numeric copy_numeric(numeric inp):
    return inp

cdef matrix copy_matrix(matrix inp):
    return inp

cdef varidx copy_varidx(varidx inp):
    return inp

cdef spinidx copy_spinidx(spinidx inp):
    return inp

cdef idx copy_idx(idx inp):
    return inp

cdef map_function copy_map_function(map_function inp):
    return inp
#### Arithmatic operator functions for Ex object ########
cdef Ex Add(Ex lh, Ex rh):
    return ex_to_Ex(lh._this+rh._this)
cdef Ex Sub(Ex lh, Ex rh):
    return ex_to_Ex((lh._this-rh._this))
cdef Ex Mul(Ex lh, Ex rh):
    return ex_to_Ex((lh._this*rh._this))
cdef Ex Div(Ex lh, Ex rh):
    return ex_to_Ex((lh._this/rh._this))
cdef Ex Pow(Ex base, Ex exponent):
    return ex_to_Ex(_pow(base._this,exponent._this))

####################################################################################################
                                ###### add object ###############
####################################################################################################
cdef class add(Ex):
    cdef _add _thisAdd
    def __init__(self,lh,rh=None):
        super().__init__()
        if rh is not None:
            self._thisAdd = _add(py_to_ex(lh),py_to_ex(rh))
            self._this = self._thisAdd.eval()
        elif isinstance(lh,list):
            self._thisAdd = _add(list_to_exvector(lh))
            self._this = self._thisAdd.eval()
        else:
            raise Exception("unsupported operand.")


####################################################################################################
                                ###### mul object ###############
####################################################################################################
cdef class mul(Ex):
    cdef _mul _thisMul
    def __init__(self,lh,rh=None):
        super().__init__()
        if rh is not None:
            self._thisMul = _mul(py_to_ex(lh),py_to_ex(rh))
            self._this = self._thisMul.eval()
        elif isinstance(lh,list):
            self._thisMul = _mul(list_to_exvector(lh))
            self._this = self._thisMul.eval()
        else:
            raise Exception("unsupported operand.")

####################################################################################################
                                ###### power object ###############
####################################################################################################
cdef class power(Ex):
    cdef _power _thisPower
    def __init__(self,base,exponent):
        super().__init__()
        self._thisPower = _power(py_to_ex(base),py_to_ex(exponent))
        self._this = self._thisPower.eval()

#cpdef Ex add(lh, rh):
#    return ex_to_Ex((py_to_ex(lh)+py_to_ex(rh)))
cpdef Ex sub(lh,rh):
    return ex_to_Ex((py_to_ex(lh)-py_to_ex(rh)))
#cpdef Ex mul(lh, rh):
#    return ex_to_Ex((py_to_ex(lh)*py_to_ex(rh)))
cpdef Ex div(lh, rh):
    return ex_to_Ex((py_to_ex(lh)/py_to_ex(rh)))
cpdef Ex pow(base, exponent):
    return ex_to_Ex(_pow(py_to_ex(base),py_to_ex(exponent)))
#cpdef Ex power(base, exponent):
#    return ex_to_Ex(_pow(py_to_ex(base),py_to_ex(exponent)))

####################################################################################################
                                ###### wildcard object ###############
####################################################################################################
cdef class wildcard(Ex):
    cdef _wildcard _thisWildcard
    def __init__(self,int label):
        super().__init__()
        self._thisWildcard = _wildcard(label)
        self._this = self._thisWildcard.eval()
    def get_label(self):
        return self._thisWildcard.get_label()


#### wild() ###############################
cpdef Ex wild(int label):
    generator.symRegister(_wild(<unsigned>label))
    return ex_to_Ex(_wild(<unsigned>label))
cpdef bool haswild(Ex expr):
    return _haswild(expr._this)

#### find() ###############
cdef bool _find(Ex inp, Ex pattern,set found):
    cdef exset clt
    cdef bool ret= inp._this.find(pattern._this,clt)
    itr=clt.begin()
    while itr!=clt.end():
        found.add(ex_to_Ex(deref(itr)))
        inc(itr)
    return ret

##### match() ###############
cdef bool _match(Ex inp, Ex pattern,dict repls):
    cdef exmap clt
    cdef bool ret= inp._this.match(pattern._this,clt)
    itr=clt.begin()
    while itr!=clt.end():
        repls[ex_to_Ex(deref(itr).first)]=ex_to_Ex(deref(itr).second)
        inc(itr)
    return ret

##### to_rational() ###############
cdef Ex _to_rational(Ex inp,dict repls):
    cdef exmap clt
    cdef ex ret= inp._this.to_rational(clt)
    itr=clt.begin()
    while itr!=clt.end():
        generator.symRegister(deref(itr).first)
        repls[ex_to_Ex(deref(itr).first)]=ex_to_Ex(deref(itr).second)
        inc(itr)
    return ex_to_Ex(ret)

##### to_polynomial() ###############
cdef Ex _to_polynomial(Ex inp,dict repls):
    cdef exmap clt
    cdef ex ret= inp._this.to_polynomial(clt)
    itr=clt.begin()
    while itr!=clt.end():
        generator.symRegister(deref(itr).first)
        repls[ex_to_Ex(deref(itr).first)]=ex_to_Ex(deref(itr).second)
        inc(itr)
    return ex_to_Ex(ret)

#### unitcontprim ########################
cdef unitcontprim(Ex inp, Ex x):
    cdef:
        ex uclt
        ex cclt
        ex pclt
    inp._this.unitcontprim(x._this,uclt, cclt, pclt)
    return (ex_to_Ex(uclt),ex_to_Ex(cclt),ex_to_Ex(pclt))

###### imaginary unit I or i ###################
cdef Ex I_ex_to_Ex():
    return ex_to_Ex(_I.eval())
I=I_ex_to_Ex()

###### Pi,Catalan,Euler constants ###################
cdef Ex Pi_ex_to_Ex():
    return ex_to_Ex(_Pi)
cdef Ex Catalan_ex_to_Ex():
    return ex_to_Ex(_Catalan)
cdef Ex Euler_ex_to_Ex():
    return ex_to_Ex(_Euler)

Pi = Pi_ex_to_Ex()
Catalan = Catalan_ex_to_Ex()
Euler = Euler_ex_to_Ex()

###### set and get number of digits #######################
cpdef set_digits(digits):
    _set_digits(<long>digits)

cpdef get_digits():
     return _get_digits()
###### virtual map_function #####################
cdef class map_function:
    cdef imap_function* _thisptr
    def __cinit__(self):
       self._thisptr = new imap_function(<cpy_ref.PyObject*>self)
    def __dealloc__(self):
       if self._thisptr:
           del self._thisptr

cdef public api:
    cdef ex cy_call_func(object self, ex a, char* method, int *error):
        try:
            func = getattr(self, method.decode('UTF-8'));
            error[0] = 0
            return copy_Ex(func(ex_to_Ex(a)))._this
        except AttributeError:
            error[0] = 1
#        else:
#            error[0] = 0
#            return copy_Ex(func(ex_to_Ex(a)))._this



####################################################################################################
                                ###### return_type_t struct ###############
####################################################################################################
cdef class return_type_t:
    cdef _return_type_t _thisReturn_type_t
    def __eq__(self,return_type_t other):
        return self._thisReturn_type_t==other._thisReturn_type_t
    def __ne__(self,return_type_t other):
        return self._thisReturn_type_t!=other._thisReturn_type_t
    def __lt__(self,return_type_t other):
        return self._thisReturn_type_t<other._thisReturn_type_t


####################################################################################################
                                ###### Ex object ###############
####################################################################################################

cdef class Ex:
    ##### Member variable ########
    cdef ex _this
    cdef list _slist
    cdef symbol_assumptions _assumptions
    cdef Ex_iterator itr
    ####### Constructor #######
    def __init__(self,s=None,symbol_assumptions symboltype=complex):
        if type(self) is Ex:
            if s is None:
                self._this=ex()
            elif isinstance(s, str):
                s = s.replace(' ','')
                if s.find(",")==-1:
                    if isinstance(s,str):
                        if s[0]!="\\":
                            if symboltype==complex:
                                self._this= generator.exGenerator(s.encode("UTF-8"),_symbol_assumptions._symbol_assumptions_symbol,False)
                            elif symboltype==real:
                                self._this= generator.exGenerator(s.encode("UTF-8"),_symbol_assumptions._realsymbol,False)
                            elif symboltype==positive:
                                self._this= generator.exGenerator(s.encode("UTF-8"),_symbol_assumptions._possymbol,False)
                        else:
                            if symboltype==complex:
                                self._this= generator.exGenerator(s[1:].encode("UTF-8"),_symbol_assumptions._symbol_assumptions_symbol,True)
                            elif symboltype==real:
                                self._this= generator.exGenerator(s[1:].encode("UTF-8"),_symbol_assumptions._realsymbol,True)
                            elif symboltype==positive:
                                self._this= generator.exGenerator(s[1:].encode("UTF-8"),_symbol_assumptions._possymbol,True)
                else:
                    self._slist=s.split(",")## __iter__ will call
                    self._assumptions=symboltype##  with these variables
            elif isinstance(s,int):
                self._this=ex(<int>s)
            elif isinstance(s,float):
                self._this=_numeric_to_ex(string_to__numeric(str(s).encode("UTF-8")))

    ##check the assumption of a symbol###
    def assumption(self):
        if self.is_symbol():
            return get_assumptions(self._this,False)
        else:
            raise Exception("assumption on the expression can not be checked.")
    #### evaluations #####
    def eval(self):
        return ex_to_Ex(self._this.eval())
    def evalf(self):
        return ex_to_Ex(self._this.evalf())
    def evalm(self):
        return ex_to_Ex(self._this.evalm())
    def eval_ncmul(self,list v):
        return ex_to_Ex(self._this.eval_ncmul(list_to_exvector(v)))
    def eval_integ(self):
        return ex_to_Ex(self._this.eval_integ())

    #info
    def info(self,info_flags inf):
        return (self._this.info(<unsigned>inf))
    def is_real(self):
        return self._this.info(<unsigned>_real)
    def is_rational(self):
        return self._this.info(<unsigned>_rational)
    def is_integer(self):
        return self._this.info(<unsigned>_integer)
    def is_crational(self):
        return self._this.info(<unsigned>_crational)
    def is_cinteger(self):
        return self._this.info(<unsigned>_cinteger)
    def is_positive(self):
        return self._this.info(<unsigned>_positive)
    def is_negative(self):
        return self._this.info(<unsigned>_negative)
    def is_nonnegative(self):
        return self._this.info(<unsigned>_nonnegative)
    def is_posint(self):
        return self._this.info(<unsigned>_posint)
    def is_negint(self):
        return self._this.info(<unsigned>_negint)
    def is_nonnegint(self):
        return self._this.info(<unsigned>_nonnegint)
    def is_even(self):
        return self._this.info(<unsigned>_even)
    def is_odd(self):
        return self._this.info(<unsigned>_odd)
    def is_prime(self):
        return self._this.info(<unsigned>_prime)

    def is_relation(self):
        return self._this.info(<unsigned>_relation)
    def is_relation_equal(self):
        return self._this.info(<unsigned>_relation_equal)
    def is_relation_not_equal(self):
        return self._this.info(<unsigned>_relation_not_equal)
    def is_relation_less(self):
        return self._this.info(<unsigned>_relation_less)
    def is_relation_less_or_equal(self):
        return self._this.info(<unsigned>info_flags_relation_less_or_equal)
    def is_relation_greater(self):
        return self._this.info(<unsigned>_relation_greater)
    def is_relation_greater_or_equal(self):
        return self._this.info(<unsigned>_relation_greater_or_equal)

    def is_integer_polynomial(self):
        return self._this.info(<unsigned>_integer_polynomial)
    def is_cinteger_polynomial(self):
        return self._this.info(<unsigned>_cinteger_polynomial)
    def is_rational_polynomial(self):
        return self._this.info(<unsigned>_rational_polynomial)
    def is_crational_polynomial(self):
        return self._this.info(<unsigned>_crational_polynomial)
    def is_rational_function(self):
        return self._this.info(<unsigned>_rational_function)

    # operand access
    def nops(self):
        return (self._this.nops())
    def op(self,int i):
        return ex_to_Ex(self._this.op(<size_t>i))
    def let_op(self,int i):
        return ex_to_Ex(self._this.let_op(<size_t>i))
    def lhs(self):
        return ex_to_Ex(self._this.lhs())
    def rhs(self):
        return ex_to_Ex(self._this.rhs())

#    # function for complex expressions
    def conjugate(self):
        return ex_to_Ex(self._this.conjugate())
    def real_part(self):
        return ex_to_Ex(self._this.real_part())
    def imag_part(self):
        return ex_to_Ex(self._this.imag_part())

#    # pattern matching
    def has(self,Ex pattern, has_options opt=has_nooptions):
        return self._this.has(pattern._this,opt)
#        if opt is has_nooptions:
#            return self._this.has(pattern._this,0)
#        else:
#            return self._this.has(pattern._this,_has_algebraic)
    def find(self,Ex pattern, set found):
        return _find(self,pattern,found)
    def match(self,Ex pattern, dict repls=None):
        if repls is None:
           return self._this.match(pattern._this)
        else:
            return _match(self,pattern,repls)

    # substitutions
    def subs(self,ps1, subs_options opt =subs_nooptions):
#        if isinstance(ps1,list) and isinstance(ps2,list):
#            return ex_to_Ex(self._this.subs(list_to_cpplist(ps1),list_to_cpplist(ps2),opt))
#        if isinstance(ps1,dict):
        return ex_to_Ex(self._this.subs(dict_to_exmap(ps1),opt))

    # function mapping
#    def (self,map_function mapfunc):
#        return ex_to_Ex(self._this.map(deref(mapfunc)))
    def map(self,argu):
        if isinstance(argu,map_function):
            return ex_to_Ex(self._this.map(deref(copy_map_function(argu)._thisptr)))
        elif isinstance(argu,Ex):
            return ex_to_Ex(f_map((self._this),copy_Ex(argu)._this))
        else:
            raise Exception("invalid argument.")

    # visitors and tree traversal
#        void accept(visitor & v)
#        void traverse_preorder(visitor & v)
#        void traverse_postorder(visitor & v)
#        void traverse(visitor & v)

    # degree/coeff
    def is_polynomial(self,Ex vars):
        return self._this.is_polynomial(vars._this)
    def degree(self,Ex s):
        return self._this.degree(s._this)
    def ldegree(self,Ex s):
        return self._this.ldegree(s._this)
    def coeff(self,Ex s, int n = 1):
        return ex_to_Ex(self._this.coeff(s._this,n))
    def lcoeff(self,Ex s):
        return ex_to_Ex(self._this.lcoeff(s._this))
    def tcoeff(self,Ex s):
        return ex_to_Ex(self._this.tcoeff(s._this))

    # expand/collect
    def power_expand(self,expand_options opt=expand_nooptions):
        return ex_to_Ex(self._this.power_expand(opt))
    def expand(self,expand_options opt=expand_nooptions):
        return ex_to_Ex(expandflint(self._this,opt))
#        if opt is expand_nooptions:
#            return ex_to_Ex(self._this.expand(0))
#        elif opt is expand_options.expand_function_args:
#            return ex_to_Ex(expandflint(self._this,_expand_options._expand_function_args))
#        elif opt is expand_indexed:
#            return ex_to_Ex(expandflint(self._this,_expand_options._expand_indexed))
#        elif opt is expand_rename_idx:
#            return ex_to_Ex(expandflint(self._this,_expand_options._expand_rename_idx))
#        elif opt is expand_transcendental:
#            return ex_to_Ex(expandflint(self._this,_expand_options._expand_transcendental))
    def collect(self,Ex s, bool distributed = False):
        return ex_to_Ex(self._this.collect(s._this,distributed))

    # differentiation and series expansion
    def diff(self,Ex s, int nth = 1):
        return ex_to_Ex(self._this.diff(ex_to[_symbol](s._this),nth))
    def series(self,Ex r, int order, series_options opt=series_nooptions):
        return ex_to_Ex(self._this.series(r._this,order,opt))
#        if opt is series_nooptions:
#            return ex_to_Ex(self._this.series(r._this,order,0))
#        else:
#            return ex_to_Ex(self._this.series(r._this,order,_suppress_branchcut))

    # rational functions
    def normal(self):
        return ex_to_Ex(self._this.normal())
    def to_rational(self,dict repl):
        return _to_rational(self,repl)
    def to_polynomial(self,dict repl):
        return _to_polynomial(self,repl)
    def numer(self):
        return ex_to_Ex(self._this.numer())
    def denom(self):
        return ex_to_Ex(self._this.denom())
    def numer_denom(self):
        return ex_to_Ex(self._this.numer_denom())

    # polynomial algorithms
    def unit(self,Ex x):
        return ex_to_Ex(self._this.unit(x._this))
    def content(self,Ex x):
        return ex_to_Ex(self._this.content(x._this))
    def integer_content(self):
        return _numeric_to_numeric(self._this.integer_content())
    def primpart(self,Ex x, Ex cont=None):
        if cont is None:
            return ex_to_Ex(self._this.primpart(x._this))
        else:
            return ex_to_Ex(self._this.primpart(x._this,cont._this))
    def unitcontprim(self,Ex x):
        return unitcontprim(self,x)
    def smod(self,numeric xi):
        return ex_to_Ex(self._this.smod(xi._thisNumeric))
    def max_coefficient(self):
        return _numeric_to_numeric(self._this.max_coefficient())

    # indexed objects
    def get_free_indices(self):
        return exvector_to_Ex_list(self._this.get_free_indices())
    def get_all_dummy_indices(self):
        return get_all_dummy_indices(self)
    def get_all_dummy_indices_safely(self):
        return get_all_dummy_indices_safely(self)
    def simplify_indexed(self,scalar_products sp=None):
        if sp is None:
            return ex_to_Ex(self._this.simplify_indexed(0))
        else:
            return ex_to_Ex(self._this.simplify_indexed(sp._thisSp,0))

    # pseries
    def get_var(self):
        if is_a[_pseries](self._this):
            return ex_to_Ex(ex_to[_pseries](self._this).get_var())
        else:
            raise Exception("Ex object is not pseries.")

    def get_point(self):
        if is_a[_pseries](self._this):
            return ex_to_Ex(ex_to[_pseries](self._this).get_point())
        else:
            raise Exception("Ex object is not pseries.")
    def convert_to_poly(self,bool no_order = False):
        if is_a[_pseries](self._this):
            return ex_to_Ex(ex_to[_pseries](self._this).convert_to_poly(no_order))
        else:
            raise Exception("Ex object is not pseries.")

    def is_compatible_to(self, pseries other):
        if is_a[_pseries](self._this):
            return (ex_to[_pseries](self._this).is_compatible_to((other._thisPseries)))
        else:
            raise Exception("Ex object is not pseries.")
#    def is_zero(self):
#        if is_a[_pseries](self._this):
#            return (ex_to[_pseries](self._this).is_zero())
#        else:
#            raise Exception("Ex object is not pseries.")

    def is_terminating(self):
        if is_a[_pseries](self._this):
            return (ex_to[_pseries](self._this).is_terminating())
        else:
            raise Exception("Ex object is not pseries.")

    def coeffop(self,int i):
        if is_a[_pseries](self._this):
            return ex_to_Ex(ex_to[_pseries](self._this).coeffop(<size_t>i))
        else:
            raise Exception("Ex object is not pseries.")
    def exponop(self,int i):
        if is_a[_pseries](self._this):
            return ex_to_Ex(ex_to[_pseries](self._this).exponop(<size_t>i))
        else:
            raise Exception("Ex object is not pseries.")

    def add_series(self, pseries other):
        if is_a[_pseries](self._this):
            return ex_to_Ex(ex_to[_pseries](self._this).add_series(other._thisPseries))
        else:
            raise Exception("Ex object is not pseries.")
    def mul_const(self, other):
        tem = ex_to_Ex(py_to_ex(other))
        if is_a[_pseries](self._this) and is_a[_numeric](tem._this):
            return ex_to_Ex(ex_to[_pseries](self._this).mul_const(ex_to[_numeric](tem._this)))
        else:
            raise Exception("argument is not numeric.")
    def mul_series(self, pseries other):
        if is_a[_pseries](self._this):
            return ex_to_Ex(ex_to[_pseries](self._this).mul_series(other._thisPseries))
        else:
            raise Exception("Ex object is not pseries.")
    def power_const(self,p, int deg):
        tem = ex_to_Ex(py_to_ex(p))
        if is_a[_pseries](self._this) and is_a[_numeric](tem._this):
            return ex_to_Ex(ex_to[_pseries](self._this).power_const(ex_to[_numeric](tem._this),<int>deg))
        else:
            raise Exception("argument is not numeric.")
    def shift_exponents(self,int deg):
        if is_a[_pseries](self._this):
            return _pseries_to_pseries(ex_to[_pseries](self._this).shift_exponents(<int>deg))
        else:
            raise Exception("Ex object is not pseries.")

    # comparison
    def compare(self,Ex  other):
        return (self._this.compare(other._this))
    def is_equal(self,Ex  other):
        return (self._this.is_equal(other._this))
    def is_zero(self):
        return (self._this.is_zero())
    def is_zero_matrix(self):
        return (self._this.is_zero_matrix())

    # symmetry
    def symmetrize(self):
        return ex_to_Ex(self._this.symmetrize())
    #ex symmetrize(const lst & l):
    def antisymmetrize(self):
        return ex_to_Ex(self._this.antisymmetrize())
    #ex antisymmetrize(const lst & l):
    def symmetrize_cyclic(self):
        return ex_to_Ex(self._this.symmetrize_cyclic())
    #ex symmetrize_cyclic(const lst & l):

    # noncommutativity
    def return_type(self):
        return (self._this.return_type())
    def return_type_tinfo(self):
        return _return_type_t_to_return_type_t(self._this.return_type_tinfo())

    def gethash(self):
        return (self._this.gethash())


    ####### conversion to numeric value ####
    def to_int(self):
        if is_a[_numeric](self._this):
            return ex_to[_numeric](self._this).to_long()
        else:
            raise Exception("it is not numeric expression.")
    def to_double(self):
        if is_a[_numeric](self._this):
            return ex_to[_numeric](self._this).to_double()
        else:
            raise Exception("it is not numeric expression.")

    ######## some usefull functions #######
    def collect_all(self,Ex var, bool distributed=False):
        return ex_to_Ex(collectAll(self._this,var._this,distributed))
    def get_symbols(self):
        return exset_to_lst(get_symbols(self._this))

    ####### is_a[]() functions ############
    def is_a(self,mathematical_object_name):
        if isinstance(mathematical_object_name,str):
            return string((ex_to[basic](self._this)).class_name())==<string>(mathematical_object_name.encode("UTF-8"))
        else:
            return string((ex_to[basic](self._this)).class_name())==(<string>(mathematical_object_name.__name__.encode("UTF-8")))
        #return False
    def is_ex_the_function(self,mathematical_function_name):
        if is_a[_function](self._this):
            if isinstance(mathematical_function_name,str):
                return string(ex_to[_function](self._this).get_name())==<string>(mathematical_function_name.encode("UTF-8"))
            else:
                return string(ex_to[_function](self._this).get_name())==(<string>(mathematical_function_name.__name__.encode("UTF-8")))
        else:
            return False
    def is_add(self):
        return is_a[_add](self._this)
    def is_mul(self):
        return is_a[_mul](self._this)
    def is_power(self):
        return is_a[_power](self._this)
    def is_symbol(self):
        return is_a[_symbol](self._this)
    def is_numeric(self):
        return is_a[_numeric](self._this)
    def is_relational(self):
        return is_a[_relational](self._this)
    def is_matrix(self):
        return is_a[_matrix](self._this)
    def is_idx(self):
        return is_a[_idx](self._this)
    def is_varidx(self):
        return is_a[_varidx](self._this)
    def is_spinidx(self):
        return is_a[_spinidx](self._this)
    def is_indexed(self):
        return is_a[_indexed](self._this)
    def is_functions(self):
        return is_a[_functions](self._this)
    def is_Limit(self):
        return is_a[_Limit](self._this)
    def is_Diff(self):
        return is_a[_Diff](self._this)
    def is_Integrate(self):
        return is_a[_Integrate](self._this)
    def is_pseries(self):
        return is_a[_pseries](self._this)
    def is_number(self):
        return is_number(self._this)
    ######## conversions ###############
    def to_string(self):
        return to_string(self._this).decode("UTF-8")
    def to_latex_string(self):
        return to_latex_string(self._this).decode("UTF-8")
    def to_python_string(self):
        return to_python_string(self._this).decode("UTF-8")#.replace("^","**")
    def to_numeric(self):
        if is_a[_numeric](self._this):
            return _numeric_to_numeric(ex_to[_numeric](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_relational(self):
        if is_a[_relational](self._this):
            return _relational_to_relational(ex_to[_relational](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_lst(self):
        if is_a[_lst](self._this):
            return _lst_to_lst(ex_to[_lst](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_matrix(self):
        if is_a[_matrix](self._this):
            return _matrix_to_matrix(ex_to[_matrix](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_idx(self):
        if is_a[_idx](self._this):
            return _idx_to_idx(ex_to[_idx](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_varidx(self):
        if is_a[_varidx](self._this):
            return _varidx_to_varidx(ex_to[_varidx](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_spinidx(self):
        if is_a[_spinidx](self._this):
            return _spinidx_to_spinidx(ex_to[_spinidx](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_indexed(self):
        if is_a[_indexed](self._this):
            return _indexed_to_indexed(ex_to[_indexed](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_pseries(self):
        if is_a[_pseries](self._this):
            return _pseries_to_pseries(ex_to[_pseries](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_functions(self):
        if is_a[_functions](self._this):
            return _functions_to_functions(ex_to[_functions](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_Diff(self):
        if is_a[_Diff](self._this):
            return _Diff_to_Diff(ex_to[_Diff](self._this))
        else:
            raise Exception("Invalid conversion.")
    def to_Integrate(self):
        if is_a[_Integrate](self._this):
            return _Integrate_to_Integrate(ex_to[_Integrate](self._this))
        else:
            raise Exception("Invalid conversion.")


    ########## Arithmatic operators ####################
    def __add__(lh,rh):
        if isinstance(lh,Ex) and isinstance(rh,Ex):
            return Add(lh,rh)
        elif (isinstance(rh,int) or isinstance(rh,float)):
            return Add(lh,Ex(rh))
    def __radd__(self,lh):
        if isinstance(lh,Ex):
            return Add(lh,self)
        elif (isinstance(lh,int) or isinstance(lh,float)):
            return Add(Ex(lh),self)
    def __sub__(lh,rh):
        if isinstance(lh,Ex) and isinstance(rh,Ex):
            return Sub(lh,rh)
        elif (isinstance(rh,int) or isinstance(rh,float)):
            return Sub(lh,Ex(rh))
    def __rsub__(self,lh):
        if isinstance(lh,Ex):
            return Sub(lh,self)
        elif (isinstance(lh,int) or isinstance(lh,float)):
            return Sub(Ex(lh),self)
    def __mul__(self,rh):
        if isinstance(rh,Ex):
            return Mul(self,rh)
        elif (isinstance(rh,int) or isinstance(rh,float)):
            return Mul(self,Ex(rh))
    def __rmul__(self,lh):
        if isinstance(lh,Ex):
            return Mul(lh,self)
        elif (isinstance(lh,int) or isinstance(lh,float)):
            return Mul(Ex(lh),self)
    def __truediv__(self,de):
        if isinstance(de,Ex):
            return Div(self,de)
        elif (isinstance(de,int) or isinstance(de,float)):
            return Div(self,Ex(de))
    def __rtruediv__(self,nu):
        if isinstance(nu,Ex):
            return Div(nu,self)
        elif (isinstance(nu,int) or isinstance(nu,float)):
            return Div(Ex(nu),self)
    def __pow__(self,exponent):
        if isinstance(exponent,Ex):
            return Pow(self,exponent)
        elif (isinstance(exponent,int) or isinstance(exponent,float)):
            return Pow(self,Ex(exponent))
    def __rpow__(self,base):
        if isinstance(base,Ex):
            return Pow(base,self)
        elif (isinstance(base,int) or isinstance(base,float)):
            return Pow(Ex(base),self)
    def __neg__(self):
        return ex_to_Ex(-self._this)
    def __pos__(self):
        return ex_to_Ex(self._this)

    ######## Iterator #####################
    def set_iter(self,Ex_iterator itr=const_iterator):
        self.itr=itr

    def __iter__(self):
        if self._slist is None:
            if self.itr is None:
               self.itr = const_iterator
            if self.itr is const_iterator:
                it=self._this.begin()
                while it!=self._this.end():
                    yield ex_to_Ex(deref(it))
                    inc(it)
            elif self.itr is const_preorder_iterator:
                itpre=self._this.preorder_begin()
                while itpre!=self._this.preorder_end():
                    yield ex_to_Ex(deref(itpre))
                    inc(itpre)
            elif self.itr is const_postorder_iterator:
                itpost=self._this.postorder_begin()
                while itpost!=self._this.postorder_end():
                    yield ex_to_Ex(deref(itpost))
                    inc(itpost)
        else:
            for i in self._slist:
                yield Ex(i,self._assumptions)
    def __reversed__(self):
        it=self._this.end()
        while it!=self._this.begin():
            dec(it)
            yield ex_to_Ex(deref(it))
    ###########print ############################
    def __str__(self):
        return to_python_string(self._this).decode("UTF-8")
    ############ Real time printing #################
    def __repr__(self):
        #return self._repr_latex_()
        return (to_python_string(self._this).decode("UTF-8"))
#    def _repr_pretty_(self,p,cycle):
#        return p.text((to_string(self._this).decode("UTF-8")))
    def _repr_latex_(self):
        return r"$%s$"%to_latex_string(self._this).decode("UTF-8")

    ########### equality check ###################
    def __eq__(self,other):
        if isinstance(other,int) or isinstance(other,float) or isinstance(other,str):
            return (self._this).is_equal(Ex(other)._this)
        elif isinstance(other,Ex):
            return (self._this.is_equal(copy_Ex(other)._this))
        else:
            raise Exception("Equality check failure.")
    def __hash__(self):
        return self._this.gethash()

    def __ne__(self,other):
        return self._this!=py_to_ex(other)
    def __lt__(self,other):
        return self._this<py_to_ex(other)
    def __gt__(self,other):
        return self._this>py_to_ex(other)
    def __le__(self,other):
        return self._this<=py_to_ex(other)
    def __ge__(self,other):
        return self._this>=py_to_ex(other)

    def __getitem__(self, int index): ##[] operator for slicing lst object
        if index >= (self._this).nops() or index<0:
            raise Exception("index is out of range or negative.")
        return ex_to_Ex((self._this)[<size_t>index])
    def __setitem__(self, int index,other): ##[] operator for setting lst object
        if index >= (self._this).nops() or index<0:
            raise Exception("index is out of range or negative.")
        (self._this)[<size_t>index]=py_to_ex(other)

    #####some new member functions#####################
    def evaluate(self):
        return ex_to_Ex(_evaluate(self._this))
    def apply_partial_integration(self, int partial_num):
        return ex_to_Ex(_apply_partial_integration_on_ex(self._this,partial_num))

    def integrate(self, Ex var_, int partial_num=-1, l_=None, u_=None):
        if l_ is None and u_ is None:
            return ex_to_Ex(_integrate(self._this, var_._this, partial_num))
        else:
            return ex_to_Ex(_integrate(self._this,var_._this,py_to_ex(l_),py_to_ex(u_),partial_num))

    def limit(self,Ex z, z0, str dir="+-"):
        return ex_to_Ex(_limit(self._this,z._this, py_to_ex(z0), dir.encode("UTF-8")))

    def factor(self, factor_options opt=factor_options_polynomial):
        return ex_to_Ex(_factor(self._this,opt))
#        if opt is factor_nooptions:
#            return ex_to_Ex(_factor(self._this,0))
#        elif opt is polynomial:
#            return ex_to_Ex(_factor(self._this,_factor_options._polynomial))
#        else:
#            return ex_to_Ex(_factor(self._this,_factor_options._all))

    def collect_common_factor(self):
        return ex_to_Ex(Collect_common_factor(self._this))

    #simplifications
    def Simplify(self, simplify_options rules=simplify_nooptions):
        if rules is simplify_nooptions:
            return ex_to_Ex(_fullSimplify(self._this,_funcSimp))
        else:
            return ex_to_Ex(_fullSimplify(self._this,rules))
    def simplify(self, simplify_options rules=simplify_nooptions):
        if rules is simplify_nooptions:
            return ex_to_Ex(_simplify(self._this,_funcSimp))
        else:
            return ex_to_Ex(_simplify(self._this,rules))
    def fullsimplify(self, simplify_options rules=simplify_nooptions):
        if rules is simplify_nooptions:
            return ex_to_Ex(_fullsimplify(self._this,_funcSimp))
        else:
            return ex_to_Ex(_fullsimplify(self._this,rules))

    #number theory
    def base_form(self,int b):
        return _base_form(self._this,b).decode("UTF-8")
    def divisible(self, n):
        return _divisible(self._this,py_to_ex(n))
    def next_prime(self):
        return ex_to_Ex(_next_prime(self._this))
    def random_prime(self):
        return ex_to_Ex(_random_prime(self._this))


#    // wrapper functions around member functions
cpdef size_t nops(Ex expr):
    return expr._this.nops()

cpdef lst expand(Ex expr, expand_options opt=expand_nooptions):
    return to_lst(expandflint(expr._this,opt))

cpdef Ex factor(Ex poly, factor_options opt=factor_nooptions):
    return ex_to_Ex(_factor(poly._this,opt))

cpdef Ex diff(Ex expr, Ex var , int order=1):
    return ex_to_Ex(expr._this.diff(ex_to[_symbol](var._this),order))

    #number theory
cpdef base_form(n,int b):
    return _base_form(py_to_ex(n),b).decode("UTF-8")
cpdef divisible(m, n):
    return _divisible(py_to_ex(m),py_to_ex(n))
cpdef next_prime(n):
    return ex_to_Ex(_next_prime(py_to_ex(n)))
cpdef random_prime(n):
    return ex_to_Ex(_random_prime(py_to_ex(n)))

####################################################################################################
                                ###### symbol object ###############
####################################################################################################
cdef class symbol(Ex):
    cdef _symbol _thisSymbol
    def __init__(self,str symb, str latexname=None):
        super().__init__()
        if latexname is None:
            self._thisSymbol = _symbol(symb.encode('utf-8'))
        else:
            self._thisSymbol = _symbol(symb.encode('utf-8'),latexname.encode('utf-8'))
        self._this = self._thisSymbol.eval()

####################################################################################################
                                ###### numeric object ###############
####################################################################################################
cdef class numeric(Ex):
    cdef _numeric _thisNumeric
    def __init__(self,num=None):
        super().__init__()
        if num is not None:
            if isinstance(num,int):
                self._thisNumeric=_numeric(<int>num)
            elif isinstance(num,str):
                self._thisNumeric=_numeric(<str>num.encode('UTF-8'))
            self._this=self._thisNumeric.eval()

    def csgn(self):
        return self._thisNumeric.csgn()
    def compare(self,numeric other):
        self._thisNumeric.compare(other._thisNumeric)
#    def is_equal(self,numeric other):
#        self._thisNumeric.is_equal(other._thisNumeric)
#    def is_zero(self):
#        self._thisNumeric.is_zero()
#    def is_positive(self):
#        self._thisNumeric.is_positive()
#    def is_negative(self):
#        self._thisNumeric.is_negative()
#    def is_integer(self):
#        self._thisNumeric.is_integer()
#    def is_pos_integer(self):
#        self._thisNumeric.is_pos_integer()
#    def is_nonneg_integer(self):
#        self._thisNumeric.is_nonneg_integer()
#    def is_even(self):
#        self._thisNumeric.is_even()
#    def is_odd(self):
#        self._thisNumeric.is_odd()
#    def is_prime(self):
#        self._thisNumeric.is_prime()
#    def is_rational(self):
#        self._thisNumeric.is_rational()
#    def is_real(self):
#        self._thisNumeric.is_real()
#    def is_cinteger(self):
#        self._thisNumeric.is_cinteger()
#    def is_crational(self):
#        self._thisNumeric.is_crational()

####################################################################################################
                                ###### constant object ###############
####################################################################################################
cdef class constant(Ex):
    pass

####################################################################################################
                                ###### normal.h ###############
####################################################################################################
cpdef Ex quo(Ex a, Ex b, Ex x, bool check_args = True):
    return ex_to_Ex(_quo(a._this, b._this, x._this, check_args))

cpdef Ex rem(Ex a, Ex b, Ex x, bool check_args = True):
    return ex_to_Ex(_rem(a._this, b._this, x._this, check_args))

cpdef Ex decomp_rational(Ex a, Ex x):
    return ex_to_Ex(_decomp_rational(a._this, x._this))

cpdef Ex prem(Ex a, Ex b, Ex x, bool check_args = True):
    return ex_to_Ex(_prem(a._this, b._this, x._this, check_args))

cpdef Ex sprem(Ex a, Ex b, Ex x, bool check_args = True):
    return ex_to_Ex(_sprem(a._this, b._this, x._this, check_args))

cpdef divide(Ex a, Ex b):
    cdef:
        ex qr
        bool ret = (_divide(a._this, b._this, qr, True))
    if ret:
        return ret,ex_to_Ex(qr)
    else:
        return (False,)

cpdef gcd(list l):
    return ex_to_Ex(Gcd(list_to__lst(l)))

cpdef Ex lcm(list l):
    return ex_to_Ex(Lcm(list_to__lst(l)))

cpdef Ex sqrfree(Ex a, lst l = lst([])):
    return ex_to_Ex(_sqrfree(a._this, l._thisLst))

cpdef Ex sqrfree_parfrac(Ex a, Ex x):
    return ex_to_Ex(_sqrfree_parfrac(a._this, ex_to[_symbol](x._this)))

cpdef Ex apart(Ex a, Ex x):
    return ex_to_Ex(_apart(a._this, ex_to[_symbol](x._this)))

cpdef Ex resultant(Ex  e1, Ex  e2, Ex  s):
    return ex_to_Ex(_resultant(e1._this, e2._this, s._this))

####################################################################################################
                                ###### relational object ###############
####################################################################################################
cdef class relational(Ex):
    cdef _relational _thisRelational
    def __init__(self,l=None, r=None, operators oper=equal):
        super().__init__()
        if l is not None:
            if oper is equal:
                self._thisRelational=_relational(py_to_ex(l),py_to_ex(r),_equal)
            elif oper is not_equal:
                self._thisRelational=_relational(py_to_ex(l),py_to_ex(r),_not_equal)
            elif oper is less:
                self._thisRelational=_relational(py_to_ex(l),py_to_ex(r),_less)
            elif oper is less_or_equal:
                self._thisRelational=_relational(py_to_ex(l),py_to_ex(r),_less_or_equal)
            elif oper is greater:
                self._thisRelational=_relational(py_to_ex(l),py_to_ex(r),_greater)
            elif oper is greater_or_equal:
                self._thisRelational=_relational(py_to_ex(l),py_to_ex(r),_greater_or_equal)

            self._this=_relational_to_ex(self._thisRelational)


####################################################################################################
                                ###### lst object ###############
####################################################################################################
cdef class lst(Ex):
    cdef _lst _thisLst
    cdef list _thisList
    def __init__(self,list l=None):
        super().__init__()
        if l is not None:
            self._thisList = l
            self._thisLst = list_to__lst(l)
            self._this=_lst_to_ex(self._thisLst)

    def add(self):
        return ex_to_Ex(_add(list_to_exvector(self._thisList)).eval())
    def mul(self):
        return ex_to_Ex(_mul(list_to_exvector(self._thisList)).eval())
    def prepend(self,Ex b):
        prepend(self._this,b._this)
    def append(self,Ex b):
        append(self._this,b._this)
    def remove_first(self):
        remove_first(self._this)
    def remove_last(self):
        remove_last(self._this)
    def remove_all(self):
        remove_all(self._this)
    def sort(self):
        sort(self._this)
    def unique(self):
        unique(self._this)


cdef lst to_lst(ex other):
    cdef lst tem=lst(None)
    tem._this=other
    return tem
####################################################################################################
                                ###### matrix object ###############
####################################################################################################
cdef cpplist[cpplist[ex]] casEx_to_casex_list(list linp):
    cdef:
        cpplist[cpplist[ex]] clt
    for i in linp:
        clt.push_back(list_to_cpplist(i))
    return clt

cpdef matrix unit_matrix(int r, int c=0):
    if c is 0:
        return _matrix_to_matrix(ex_to[_matrix](_unit_matrix(r)))
    else:
        return _matrix_to_matrix(ex_to[_matrix](_unit_matrix(r,c)))
cpdef matrix diag_matrix(list l):
    return _matrix_to_matrix(ex_to[_matrix](_diag_matrix(list_to__lst(l))))
cpdef matrix symbolic_matrix(int r, int c, str base_name, str tex_base_name=None):
    if tex_base_name is None:
        return _matrix_to_matrix(ex_to[_matrix](_symbolic_matrix(r,c,base_name.encode("UTF-8"))))
    else:
        return _matrix_to_matrix(ex_to[_matrix](_symbolic_matrix(r,c,base_name.encode("UTF-8"),tex_base_name.encode("UTF-8"))))

cdef class matrix(Ex):
    cdef _matrix _thisMatrix
    def __init__(self,list linp=None):
        super().__init__()
        if linp is not None:
            super(matrix,self).__init__()
            self._thisMatrix= _Matrix(casEx_to_casex_list(linp))
            self._this=_matrix_to_ex(self._thisMatrix)

    def evalm(self):
        return ex_to_Ex(self._thisMatrix.evalm())
    def eval_indexed(self,Ex i):
        return ex_to_Ex(self._thisMatrix.eval_indexed(ex_to[basic](i._this)))
    def add_indexed(self,Ex own, Ex other):
        return ex_to_Ex(self._thisMatrix.add_indexed(own._this,other._this))
    def scalar_mul_indexed(self,Ex own, numeric other):
        return ex_to_Ex(self._thisMatrix.scalar_mul_indexed(own._this,other._thisNumeric))
    #bool contract_with(exvector::iterator self, exvector::iterator other, exvector & v)
    def rows(self):        #/ Get number of rows.
        return (self._thisMatrix.rows())
    def cols(self):       #/ Get number of columns.
        return (self._thisMatrix.cols())
    def add(self,matrix other):
        return _matrix_to_matrix(self._thisMatrix.add(other._thisMatrix))
    def sub(self,matrix other):
        return _matrix_to_matrix(self._thisMatrix.sub(other._thisMatrix))
    def mul(self,other):
        if isinstance(other,matrix):
            return _matrix_to_matrix(self._thisMatrix.mul(copy_matrix(other)._thisMatrix))
        elif isinstance(other,numeric):
           return _matrix_to_matrix(self._thisMatrix.mul(copy_numeric(other)._thisNumeric))
        else:
           raise TypeError
    def mul_scalar(self,other):
        return _matrix_to_matrix(self._thisMatrix.mul_scalar(py_to_ex(other)))
    def pow(self,int expn):
        return _matrix_to_matrix(self._thisMatrix.pow(ex(expn)))
    def __call__(self,int ro, int so):
        return ex_to_Ex(self._thisMatrix(<unsigned>ro,<unsigned>so))
    def __getitem__(self, index): ##[] operator for slicing matrix object
        ro,co = index
        return ex_to_Ex(self._thisMatrix(<unsigned>ro,<unsigned>co))
    def __setitem__(self, index, other): ##[] operator for setting matrix object
        ro,co = index
        self._thisMatrix.set(ro,co,py_to_ex(other))
        self._this=_matrix_to_ex(self._thisMatrix)
    def set(self,int ro, int co, other):
        self._thisMatrix.set(ro,co,py_to_ex(other))
        self._this=_matrix_to_ex(self._thisMatrix)
    def reduced_matrix(self,int r, int c):
        return _matrix_to_matrix(ex_to[_matrix](reduced_matrix(self._thisMatrix,r,c)))
    def sub_matrix(self,int r, int nr, int c, int nc):
        return _matrix_to_matrix(ex_to[_matrix](sub_matrix(self._thisMatrix,r,nr,c,nc)))
    def transpose(self):
        return _matrix_to_matrix(self._thisMatrix.transpose())
    def determinant(self,determinant_algo algo = determinant_algo_automatic):
        return ex_to_Ex(self._thisMatrix.determinant(algo))
#        if algo is determinant_algo_automatic:
#            return ex_to_Ex(self._thisMatrix.determinant(_determinant_algo._determinant_algo_automatic))
#        elif algo is determinant_algo_gauss:
#            return ex_to_Ex(self._thisMatrix.determinant(_determinant_algo._determinant_algo_gauss))
#        elif algo is determinant_algo_divfree:
#            return ex_to_Ex(self._thisMatrix.determinant(_determinant_algo._determinant_algo_divfree))
#        elif algo is determinant_algo_laplace:
#            return ex_to_Ex(self._thisMatrix.determinant(_determinant_algo._determinant_algo_laplace))
#        elif algo is determinant_algo_bareiss:
#            return ex_to_Ex(self._thisMatrix.determinant(_determinant_algo._determinant_algo_bareiss))
    def trace(self):
        return ex_to_Ex(self._thisMatrix.trace())
    def charpoly(self,Ex lambda1):
        return ex_to_Ex(self._thisMatrix.charpoly(lambda1._this))
    def inverse(self,solve_algo algo=solve_algo_automatic):
        return _matrix_to_matrix(self._thisMatrix.inverse(algo))
#        if algo is solve_algo_automatic:
#            return _matrix_to_matrix(self._thisMatrix.inverse(_solve_algo._solve_algo_automatic))
#        elif algo is solve_algo_gauss:
#            return _matrix_to_matrix(self._thisMatrix.inverse(_solve_algo._solve_algo_gauss))
#        elif algo is solve_algo_divfree:
#            return _matrix_to_matrix(self._thisMatrix.inverse(_solve_algo._solve_algo_divfree))
#        elif algo is solve_algo_bareiss:
#            return _matrix_to_matrix(self._thisMatrix.inverse(_solve_algo._solve_algo_bareiss))
    def solve(self,matrix vars, matrix rhs, solve_algo algo = solve_algo_automatic):
        return _matrix_to_matrix(self._thisMatrix.solve(vars._thisMatrix,rhs._thisMatrix,algo))
#        if algo is solve_algo_automatic:
#            return _matrix_to_matrix(self._thisMatrix.solve(vars._thisMatrix,rhs._thisMatrix,_solve_algo._solve_algo_automatic))
#        elif algo is solve_algo_gauss:
#            return _matrix_to_matrix(self._thisMatrix.solve(vars._thisMatrix,rhs._thisMatrix,_solve_algo._solve_algo_gauss))
#        elif algo is solve_algo_divfree:
#            return _matrix_to_matrix(self._thisMatrix.solve(vars._thisMatrix,rhs._thisMatrix,_solve_algo._solve_algo_divfree))
#        elif algo is solve_algo_bareiss:
#            return _matrix_to_matrix(self._thisMatrix.solve(vars._thisMatrix,rhs._thisMatrix,_solve_algo._solve_algo_bareiss))
#        elif algo is solve_algo_markowitz:
#            return _matrix_to_matrix(self._thisMatrix.solve(vars._thisMatrix,rhs._thisMatrix,_solve_algo._solve_algo_markowitz))

    def rank(self):
        return self._thisMatrix.rank()
    def rank(self, solve_algo algo = solve_algo_automatic):
        return (self._thisMatrix.rank(algo))
#        if algo is solve_algo_automatic:
#            return (self._thisMatrix.rank(_solve_algo._solve_algo_automatic))
#        elif algo is solve_algo_gauss:
#            return (self._thisMatrix.rank(_solve_algo._solve_algo_gauss))
#        elif algo is solve_algo_divfree:
#            return (self._thisMatrix.rank(_solve_algo._solve_algo_divfree))
#        elif algo is solve_algo_bareiss:
#            return (self._thisMatrix.rank(_solve_algo._solve_algo_bareiss))
#        elif algo is solve_algo_markowitz:
#            return (self._thisMatrix.rank(_solve_algo._solve_algo_markowitz))
    def is_zero_matrix(self):
        return self._thisMatrix.is_zero_matrix()

####################################################################################################
                                ###### ncmul object ###############
####################################################################################################
cdef class ncmul(Ex):
    cdef _ncmul _thisNcmul
    def __init__(self,list v):
        super().__init__()
        self._thisNcmul = _ncmul(list_to_exvector(v))
        self._this = _ncmul_to_ex(self._thisNcmul)

    def get_factors(self):
        return exvector_to_Ex_list(self._thisNcmul.get_factors())

####################################################################################################
                                ###### symmetry object ###############
####################################################################################################
cdef symmetry_type get_symmetry_type(_symmetry inp):
    cdef _symmetry_type sst=inp.get_type()
    if sst == 0:
        return symmetry_type_none
    elif sst == 1:
        return symmetry_type_symmetric
    elif sst == 2:
        return symmetry_type_antisymmetric
    elif sst == 3:
        return symmetry_type_cyclic

cdef class symmetry:
    cdef _symmetry _thisSym
    def __cinit__(self, int i=-1):
        if i is not None:
            if i==-1:
                self._thisSym=_symmetry()
            else:
                self._thisSym= _symmetry(i)
    def get_type(self):
        return get_symmetry_type(self._thisSym)
    def set_type(self, symmetry_type t):
        if t == symmetry_type_none:
            self._thisSym.set_type(_symmetry_type._none)
        elif t == symmetry_type_symmetric:
            self._thisSym.set_type(_symmetry_type._symmetric)
        elif t == symmetry_type_antisymmetric:
            self._thisSym.set_type(_symmetry_type._antisymmetric)
        elif t == symmetry_type_cyclic:
            self._thisSym.set_type(_symmetry_type._cyclic)
    def has_symmetry(self):
        return self._thisSym.has_symmetry()
    def has_nonsymmetric(self):
        return self._thisSym.has_nonsymmetric()
    def has_cyclic(self):
        return self._thisSym.has_cyclic()
#    def __dealloc__(self):
#        if self._thisptr is not NULL:
#            del self._thisptr

cdef symmetry to_symmetry(_symmetry inp):
    cdef symmetry temSym = symmetry(-1)
    temSym._thisSym=inp
    return temSym


def sy_none(symmetry c1=None, symmetry c2=None, symmetry c3=None, symmetry c4=None):
    if c1 is None or c2 is None:
        return to_symmetry(_sy_none())
    elif c3 is None:
        return to_symmetry(_sy_none(c1._thisSym,c2._thisSym))
    elif c4 is None:
        return to_symmetry(_sy_none(c1._thisSym,c2._thisSym,c3._thisSym))
    else:
        return to_symmetry(_sy_none(c1._thisSym,c2._thisSym,c3._thisSym,c4._thisSym))

def sy_symm(symmetry c1=None, symmetry c2=None, symmetry c3=None, symmetry c4=None):
    if c1 is None or c2 is None:
        return to_symmetry(_sy_symm())
    elif c3 is None:
        return to_symmetry(_sy_symm(c1._thisSym,c2._thisSym))
    elif c4 is None:
        return to_symmetry(_sy_symm(c1._thisSym,c2._thisSym,c3._thisSym))
    else:
        return to_symmetry(_sy_symm(c1._thisSym,c2._thisSym,c3._thisSym,c4._thisSym))

def sy_anti(symmetry c1=None, symmetry c2=None, symmetry c3=None, symmetry c4=None):
    if c1 is None or c2 is None:
        return to_symmetry(_sy_anti())
    elif c3 is None:
        return to_symmetry(_sy_anti(c1._thisSym,c2._thisSym))
    elif c4 is None:
        return to_symmetry(_sy_anti(c1._thisSym,c2._thisSym,c3._thisSym))
    else:
        return to_symmetry(_sy_anti(c1._thisSym,c2._thisSym,c3._thisSym,c4._thisSym))

def sy_cycl(symmetry c1=None, symmetry c2=None, symmetry c3=None, symmetry c4=None):
    if c1 is None or c2 is None:
        return to_symmetry(_sy_cycl())
    elif c3 is None:
        return to_symmetry(_sy_cycl(c1._thisSym,c2._thisSym))
    elif c4 is None:
        return to_symmetry(_sy_cycl(c1._thisSym,c2._thisSym,c3._thisSym))
    else:
        return to_symmetry(_sy_cycl(c1._thisSym,c2._thisSym,c3._thisSym,c4._thisSym))

#    // These return references to preallocated common symmetries (similar to
#    // the numeric flyweights):.
cpdef symmetry  not_symmetric():
    return to_symmetry(_not_symmetric())
cpdef symmetry  symmetric2():
    return to_symmetry(_symmetric2())
cpdef symmetry  symmetric3():
    return to_symmetry(_symmetric3())
cpdef symmetry  symmetric4():
    return to_symmetry(_symmetric4())
cpdef symmetry  antisymmetric2():
    return to_symmetry(_antisymmetric2())
cpdef symmetry  antisymmetric3():
    return to_symmetry(_antisymmetric3())
cpdef symmetry  antisymmetric4():
    return to_symmetry(_antisymmetric4())

cpdef Ex symmetrize (Ex e, list v):
    return ex_to_Ex(_symmetrize(e._this,list_to_exvector(v)))

cpdef Ex antisymmetrize (Ex e, list v):
    return ex_to_Ex(_antisymmetrize(e._this,list_to_exvector(v)))

cpdef Ex symmetrize_cyclic (Ex e, list v):
    return ex_to_Ex(_symmetrize_cyclic(e._this,list_to_exvector(v)))



####################################################################################################
                                ###### idx object ###############
####################################################################################################
cdef class idx(Ex):
    cdef _idx _thisIdx
    def __init__(self,str v=None, dim=None):
        super().__init__()
        if v is not None:
            if type(self) is idx:
                self._thisIdx= _idx(_symbol(v.encode("UTF-8")).eval(), py_to_ex(dim))
                self._this= self._thisIdx.evalf()


    def info(self,info_flags inf):
        return self._thisIdx.info(<unsigned>inf)
    def nops(self):
        return self._thisIdx.nops()
    def op(self,int i):
        return ex_to_Ex(self._thisIdx.op(i))
    def map(self,map_function f):
        return ex_to_Ex(self._thisIdx.map(f._thisptr[0]))
    def evalf(self):
        return ex_to_Ex(self._thisIdx.evalf())
    def subs(self,dict m,subs_options opt=subs_nooptions):
        return ex_to_Ex(self._thisIdx.subs(dict_to_exmap(m),opt))
    def get_value(self):
        return ex_to_Ex(self._thisIdx.get_value())
    def is_numeric(self):
        return self._thisIdx.is_numeric()
    def is_symbolic(self):
        return self._thisIdx.is_symbolic()
    def get_dim(self):
        return ex_to_Ex(self._thisIdx.get_dim())
    def is_dim_numeric(self):
        return (self._thisIdx.is_dim_numeric())
    def is_dim_symbolic(self):
        return (self._thisIdx.is_dim_symbolic())
    def replace_dim(self,new_dim):
        return _idx_to_idx(ex_to[_idx](self._thisIdx.replace_dim(py_to_ex(new_dim))))
    def minimal_dim(self,idx other):
        return ex_to_Ex(self._thisIdx.minimal_dim(other._thisIdx))

#    def __dealloc__(self):
#        if self._thisptr is not NULL:
#            del self._thisptr


####################################################################################################
                                ###### varidx object ###############
####################################################################################################
cdef class varidx(idx):
    cdef _varidx _thisVaridx
    def __init__(self,str v=None,dim=None,bool iscovariant=False):
        if v is not None:
            if type(self) is varidx:
                super(varidx,self).__init__()
                self._thisVaridx= _varidx(_symbol(v.encode("UTF-8")).eval(), py_to_ex(dim),iscovariant)
                self._this= self._thisVaridx.evalf()
                self._thisIdx = ex_to[_idx](self._this)

    def is_covariant(self):
        return (self._thisVaridx.is_covariant())
    def is_contravariant(self):
        return (self._thisVaridx.is_contravariant())
    def toggle_variance(self):
        return _varidx_to_varidx(ex_to[_varidx](self._thisVaridx.toggle_variance()))

#    def __dealloc__(self):
#        if self._thisptr is not NULL:
#            del self._thisptr

####################################################################################################
                                ###### spinidx object ###############
####################################################################################################
cdef class spinidx(varidx):
    cdef _spinidx _thisSpinidx
    def __init__(self,str v=None, dim = Ex(2), bool iscovariant = False, bool isdotted = False):
        if type(self) is spinidx and v is not None:
            self._thisSpinidx= _spinidx(_symbol(v.encode("UTF-8")).eval(), py_to_ex(dim),iscovariant,isdotted)
            self._this=self._thisSpinidx.evalf()
            self._thisVaridx=ex_to[_varidx](self._this)
            self._thisIdx=ex_to[_idx](self._this)
    def conjugate(self):
        return _spinidx_to_spinidx(ex_to[_spinidx](self._thisSpinidx.conjugate()))
    def is_dotted(self):
        return self._thisSpinidx.is_dotted()
    def is_undotted(self):
        return self._thisSpinidx.is_undotted()
    def toggle_dot(self):
        return (_spinidx_to_spinidx(ex_to[_spinidx](self._thisSpinidx.toggle_dot())))
    def toggle_variance_dot(self):
        return _spinidx_to_spinidx(ex_to[_spinidx](self._thisSpinidx.toggle_variance_dot()))
    def toggle_variance(self):
        return _spinidx_to_spinidx(ex_to[_spinidx](self._thisSpinidx.toggle_variance()))
####################################################################################################
                                ###### indexed object ###############
####################################################################################################
cdef class indexed(Ex):
    cdef _indexed _thisIndexed
    def __init__(self,Ex Exinp=None,list VaridxOrIdx=None,symmetry Sinp=None):
        super().__init__()
        if Exinp is not None:
            if Sinp is None:
                if type(VaridxOrIdx[0])==Ex:
                    self._thisIndexed= _indexed((Exinp)._this,list_to_exvector(VaridxOrIdx))
                elif type(VaridxOrIdx[0])==idx:
                    self._thisIndexed= _indexed((Exinp)._this,idx_list_to_exvector(VaridxOrIdx))
                elif type(VaridxOrIdx[0])==varidx:
                    self._thisIndexed= _indexed((Exinp)._this,varidx_list_to_exvector(VaridxOrIdx))
                elif type(VaridxOrIdx[0])==spinidx:
                    self._thisIndexed= _indexed((Exinp)._this,spinidx_list_to_exvector(VaridxOrIdx))
            else:
                if type(VaridxOrIdx[0])==Ex:
                    self._thisIndexed= _indexed((Exinp)._this,Sinp._thisSym,list_to_exvector(VaridxOrIdx))
                elif type(VaridxOrIdx[0])==idx:
                    self._thisIndexed= _indexed((Exinp)._this,Sinp._thisSym,idx_list_to_exvector(VaridxOrIdx))
                elif type(VaridxOrIdx[0])==varidx:
                    self._thisIndexed= _indexed((Exinp)._this,Sinp._thisSym,varidx_list_to_exvector(VaridxOrIdx))
                elif type(VaridxOrIdx[0])==spinidx:
                    self._thisIndexed= _indexed((Exinp)._this,Sinp._thisSym,spinidx_list_to_exvector(VaridxOrIdx))
            self._this= self._thisIndexed.eval()

    def precedence(self):
        return self._thisIndexed.precedence()
    def info(self,info_flags inf):
        return (self._thisIndexed.info(<unsigned>inf))
    def eval(self):
        return ex_to_Ex(self._thisIndexed.eval())
    def real_part(self):
        return ex_to_Ex(self._thisIndexed.real_part())
    def imag_part(self):
        return ex_to_Ex(self._thisIndexed.imag_part())
    def get_free_indices(self):
        return exvector_to_Ex_list(self._thisIndexed.get_free_indices())
    def all_index_values_are(self,info_flags inf):
        return (self._thisIndexed.all_index_values_are(<unsigned>inf))
    def get_indices(self):
        return exvector_to_Ex_list(self._thisIndexed.get_indices())
    def get_dummy_indices(self,indexed other=None):
        if other is None:
            return exvector_to_Ex_list(self._thisIndexed.get_dummy_indices())
        else:
            return exvector_to_Ex_list(self._thisIndexed.get_dummy_indices(other._thisIndexed))
    def has_dummy_index_for(self,Ex i):
        return self._thisIndexed.has_dummy_index_for(i._this)
#	/** Return symmetry properties. */
    def get_symmetry(self):
        return get_symmetry_type(ex_to[_symmetry](self._thisIndexed.get_symmetry()))

#    def __dealloc__(self):
#        if self._thisptr != NULL:
#            del self._thisptr


####################################################################################################
                                ###### scalar_products object ###############
####################################################################################################
cdef class scalar_products:
    cdef _scalar_products _thisSp
    def __init__(self):
        self._thisSp= _scalar_products()

#    def add(self,Ex v1, Ex v2, Ex sp):
#        (self._thisSp.add(v1._this,v2._this,sp._this))
    def add(self,Ex v1, Ex v2,sp,dim=None ):
        if dim is None:
            (self._thisSp.add(v1._this,v2._this,py_to_ex(sp)))
        else:
            (self._thisSp.add(v1._this,v2._this,py_to_ex(dim),py_to_ex(sp)))
    def clear(self):
        self._thisSp.clear()
    def is_defined(self,Ex v1, Ex v2, dim):
        return self._thisSp.is_defined(v1._this,v2._this,py_to_ex(dim))
    def evaluate(self,Ex v1, Ex v2, dim):
        if self._thisSp.is_defined(v1._this,v2._this,py_to_ex(dim)):
            return ex_to_Ex(self._thisSp.evaluate(v1._this,v2._this,py_to_ex(dim)))
        else:
            raise Exception("indexed objects are not defined.")

        #// utility functions
cpdef list get_all_dummy_indices(Ex e):
    return exvector_to_Ex_list(_get_all_dummy_indices(e._this))
cpdef list get_all_dummy_indices_safely(Ex e):
    return exvector_to_Ex_list(_get_all_dummy_indices_safely(e._this))
cpdef Ex rename_dummy_indices_uniquely( va, vb, bool modify_va=False):
    cdef:
        exvector tem
        Ex ret
    if isinstance(va,Ex) and isinstance(vb,Ex):
        return ex_to_Ex(_rename_dummy_indices_uniquely(copy_Ex(va)._this,copy_Ex(vb)._this))
    elif isinstance(va,list) and isinstance(vb,Ex):
        tem=list_to_exvector(va)
        ret= ex_to_Ex(_rename_dummy_indices_uniquely(tem,copy_Ex(vb)._this,modify_va))
        va=exvector_to_Ex_list(tem)
        return ret
    else:
        raise TypeError
##/** Same as above, where va and vb contain the indices of a and b and are sorted */
#ex rename_dummy_indices_uniquely(const exvector & va, const exvector & vb, Ex b)
cpdef Ex expand_dummy_sum(Ex e, bool subs_idx=False):
    return ex_to_Ex(_expand_dummy_sum(e._this,subs_idx))

####################################################################################################
                                ###### tensor ###############
####################################################################################################
cdef class tensor(Ex):
    pass
#cdef class Tensor(Ex):
#    def __init__(self):
#        if type(self) is Tensor:
#            self._this=_dynallocate[tensor]()

#cdef class Tensdelta(Tensor):
#    def __init__(self):
#        if type(self) is Tensdelta:
#            self._this=_dynallocate[tensdelta]()

#cdef class Tensmetric(Tensor):
#    def __init__(self):
#        if type(self) is Tensmetric:
#            self._this=_dynallocate[tensmetric]()

cdef class minkmetric(Ex):
    cdef _minkmetric _thisMinkmetric
    def __init__(self, bool pos_sig=False):
        super().__init__()
        if type(self) is minkmetric:
            self._thisMinkmetric=_minkmetric(pos_sig)
            self._this=_minkmetric_to_ex(self._thisMinkmetric)

#cdef class Spinmetric(Tensmetric):
#    def __init__(self):
#        if type(self) is Spinmetric:
#            self._this=_dynallocate[spinmetric]()

#cdef class Tensepsilon(Tensor):
#    def __init__(self):
#        if type(self) is Tensepsilon:
#            self._this=_dynallocate[tensepsilon]()


#    // utility functions
cpdef bool is_dummy_pair( i1, i2):
    if isinstance(i1,idx) and isinstance(i2,idx):
        return _is_dummy_pair(copy_idx(i1)._thisIdx,copy_idx(i2)._thisIdx)
    elif isinstance(i1,Ex) and isinstance(i2,Ex):
        return _is_dummy_pair(copy_Ex(i1)._this,copy_Ex(i2)._this)
    raise TypeError
def find_free_and_dummy(list v, list out_free, list out_dummy):
    cdef:
        exvector out_free_v
        exvector out_dummy_v
    _find_free_and_dummy(list_to_exvector(v),out_free_v,out_dummy_v)
    for i in exvector_to_Ex_list(out_free_v):
        out_free.append(i)
    for i in exvector_to_Ex_list(out_dummy_v):
        out_dummy.append(i)
def find_dummy_indices(list v,list out_dummy):
    cdef vector[ex] out_dummy_v
    _find_dummy_indices(list_to_exvector(v),out_dummy_v)
    for i in exvector_to_Ex_list(out_dummy_v):
        out_dummy.append(i)
def count_dummy_indices(list v):
    return _count_dummy_indices(list_to_exvector(v))
def count_free_indices(list v):
    return _count_free_indices(list_to_exvector(v))
cpdef Ex minimal_dim(dim1, dim2):
    return ex_to_Ex(_minimal_dim(py_to_ex(dim1),py_to_ex(dim2)))

cpdef Ex delta_tensor(Ex i1, Ex i2):
    return ex_to_Ex(_delta_tensor(i1._this,i2._this))

cpdef Ex metric_tensor(Ex i1, Ex i2):
    return ex_to_Ex(_metric_tensor(i1._this,i2._this))

cpdef Ex lorentz_g(Ex i1, Ex i2, bool pos_sig = False):
    return ex_to_Ex(_lorentz_g(i1._this,i2._this,pos_sig))

cpdef Ex spinor_metric(Ex i1, Ex i2):
    return ex_to_Ex(_spinor_metric(i1._this,i2._this))

cpdef Ex epsilon_tensor(Ex i1, Ex i2=None, Ex i3=None):
    if i3 is None:
        return ex_to_Ex(_epsilon_tensor(i1._this,i2._this))
    else:
        return ex_to_Ex(_epsilon_tensor(i1._this,i2._this,i3._this))

cpdef Ex lorentz_eps(Ex i1, Ex i2, Ex i3, Ex i4, bool pos_sig = False):
    return ex_to_Ex(_lorentz_eps(i1._this,i2._this,i3._this,i4._this,pos_sig))


####################################################################################################
                                ###### clifford object ###############
####################################################################################################
cdef class clifford(indexed):
    cdef _clifford _thisClifford
    def __init__(self,str b, int rl = 0,  Ex mu=None,  Ex metr=None,int comm_sign = -1):
        super().__init__()
        if mu is None and metr is None:
            self._thisClifford = _clifford(Ex(b)._this, rl)
        else:
            self._thisClifford = _clifford(Ex(b)._this, mu._this, metr._this, rl, comm_sign)
        self._thisIndexed = _clifford_to__indexed(self._thisClifford)
        self._this = _clifford_to_ex(self._thisClifford)


    def get_representation_label(self):
        return self._thisClifford.get_representation_label()
    def get_metric(self,Ex i=None, Ex j=None, symmetrised = False):
        if i is None:
            return ex_to_Ex(self._thisClifford.get_metric())
        else:
            return ex_to_Ex(self._thisClifford.get_metric(i._this,j._this,symmetrised))
    def same_metric(self,Ex other):
        return self._thisClifford.same_metric(other._this)
    def get_commutator_sign(self):
        return self._thisClifford.get_commutator_sign()


    #        // global functions

def is_clifford_tinfo(return_type_t ti):
    return _is_clifford_tinfo(ti._thisReturn_type_t)

def dirac_ONE(int rl = 0):
    return ex_to_Ex(_dirac_ONE(rl))

def clifford_unit(Ex mu, Ex metr, int rl = 0):
    return ex_to_Ex(_clifford_unit(mu._this, metr._this,rl))

def dirac_gamma(Ex mu, int rl = 0):
    return ex_to_Ex(_dirac_gamma(mu._this, rl))

def dirac_gamma5(int rl = 0):
    return ex_to_Ex(_dirac_gamma5(rl))

def dirac_gammaL(int rl = 0):
    return ex_to_Ex(_dirac_gammaL(rl))

def dirac_gammaR(int rl = 0):
    return ex_to_Ex(_dirac_gammaR(rl))

def dirac_slash(Ex e, Ex dim, int rl = 0):
    return ex_to_Ex(_dirac_slash(e._this, dim._this, rl))

def dirac_trace(Ex e, rls, Ex trONE = Ex(4)):
    if isinstance(rls,set):
        return ex_to_Ex(_dirac_trace(e._this, set_int_to_cppset(rls), trONE._this))
    elif isinstance(rls,lst):
        ex_to_Ex(_dirac_trace(e._this,copy_lst(rls)._thisLst, trONE._this))
    elif isinstance(rls,int):
        return ex_to_Ex(_dirac_trace(e._this, <int>rls, trONE._this))

def canonicalize_clifford(Ex e):
    return ex_to_Ex(_canonicalize_clifford(e._this))

def clifford_prime(Ex e):
    return ex_to_Ex(_clifford_prime(e._this))

def clifford_star_bar(Ex e, bool do_bar, int options):
    return ex_to_Ex(_clifford_star_bar(e._this,do_bar, options))

def clifford_bar(Ex e):
    return ex_to_Ex(_clifford_bar(e._this))

def clifford_star(Ex e):
    return ex_to_Ex(_clifford_star(e._this))

def remove_dirac_ONE(Ex e, int rl = 0, int options = 0):
    return ex_to_Ex(_remove_dirac_ONE(e._this, rl, options))

def clifford_max_label(Ex e, bool ignore_ONE = False):
    return _clifford_max_label(e._this, ignore_ONE)
def clifford_norm(Ex e):
    return ex_to_Ex(_clifford_norm(e._this))

def clifford_inverse(Ex e):
    return ex_to_Ex(_clifford_inverse(e._this))

def lst_to_clifford(Ex v, Ex mu,  Ex metr=None, int rl = 0):
    if metr is None:
        return ex_to_Ex(_lst_to_clifford(v._this, mu._this))
    else:
        return ex_to_Ex(_lst_to_clifford(v._this, mu._this, metr._this, rl))

def clifford_to_lst(Ex e, Ex c, bool algebraic=True):
    return _lst_to_lst(_clifford_to_lst(e._this, c._this, algebraic))

def clifford_moebius_map(Ex a, Ex b, Ex c, int rl = 0, Ex d=None, Ex v=None, Ex G=None):
    if d is None:
        return ex_to_Ex(_clifford_moebius_map(a._this, b._this, c._this, rl))
    else:
        return ex_to_Ex(_clifford_moebius_map(a._this, b._this, c._this, d._this, v._this, G._this, rl))


####################################################################################################
                                ###### color object ###############
####################################################################################################
cdef class color(indexed):
    cdef _color _thisColor
    def __init__(self, str b, int rl = 0, Ex i1=None):
        super().__init__()
        if i1 is None:
            self._thisColor = _color(Ex(b)._this, rl)
        else:
            self._thisColor = _color(Ex(b)._this, i1._this, rl)
        self._thisIndexed = _color_to__indexed(self._thisColor)
        self._this = _color_to_ex(self._thisColor)

    def get_representation_label(self):
        return self._thisClifford.get_representation_label()
#// global functions

def color_ONE(int rl = 0):
    return ex_to_Ex(_color_ONE(rl))
def color_T(Ex a, int rl = 0):
    return ex_to_Ex(_color_T(a._this, rl))
def color_f(Ex a, Ex b, Ex c):
    return ex_to_Ex(_color_f(a._this, b._this, c._this))
def color_d(Ex a, Ex b, Ex c):
    return ex_to_Ex(_color_d(a._this, b._this, c._this))
def color_h (Ex a, Ex b, Ex c):
    return ex_to_Ex(_color_h(a._this, b._this, c._this))
def color_trace(Ex e, int rl = 0, rls=None):
    if rls is None:
        return ex_to_Ex(_color_trace(e._this, rl))
    elif isinstance(rls,set):
        return ex_to_Ex(_color_trace(e._this, set_int_to_cppset(rls)))
    elif isinstance(rls,lst):
        return ex_to_Ex(_color_trace(e._this, copy_lst(rls)._thisLst))

####################################################################################################
                                ###### pseries object ###############
####################################################################################################
cdef class pseries(Ex):
    cdef _pseries _thisPseries
    def __init__(self,relational rela=None, list epvector_list=None):
        super().__init__()
        if epvector_list is not None:
            self._thisPseries=_pseries(rela._this,list_to_epvector(epvector_list))
        self._this=_pseries_to_ex(self._thisPseries)

    def get_var(self):
        return ex_to_Ex(self._thisPseries.get_var())

    def get_point(self):
        return ex_to_Ex(self._thisPseries.get_point())
    def convert_to_poly(self,bool no_order = False):
        return ex_to_Ex(self._thisPseries.convert_to_poly(no_order))

    def is_compatible_to(self,pseries other):
        return (self._thisPseries.is_compatible_to(other._thisPseries))
    def is_zero(self):
        return (self._thisPseries.is_zero())

    def is_terminating(self):
        return (self._thisPseries.is_terminating())

    def coeffop(self,int i):
        return ex_to_Ex(self._thisPseries.coeffop(<size_t>i))
    def exponop(self,int i):
        return ex_to_Ex(self._thisPseries.exponop(<size_t>i))

    def add_series(self,pseries other):
        return ex_to_Ex(self._thisPseries.add_series(other._thisPseries))
    def mul_const(self, other):
        tem = ex_to_Ex(py_to_ex(other))
        if is_a[_numeric](tem._this):
            return ex_to_Ex((self._thisPseries).mul_const(ex_to[_numeric](tem._this)))
        else:
            raise Exception("argument is not numeric.")
    def mul_series(self,pseries other):
        return ex_to_Ex(self._thisPseries.mul_series(other._thisPseries))
    def power_const(self,numeric p, int deg):
        tem = ex_to_Ex(py_to_ex(p))
        if is_a[_numeric](tem._this):
            return ex_to_Ex((self._thisPseries).power_const(ex_to[_numeric](tem._this),<int>deg))
        else:
            raise Exception("argument is not numeric.")
    def shift_exponents(self,int deg):
        return _pseries_to_pseries(self._thisPseries.shift_exponents(<int>deg))

####################################################################################################
                                ###### fail object ###############
####################################################################################################
cdef class fail(Ex):
    cdef _fail _thisFail
    def __init__(self):
        super().__init__()
        self._this = _dynallocate[_fail]()
        self._thisFail = ex_to[_fail](self._this)



####################################################################################################
                                ###### Mathematical functions ###############
####################################################################################################
#cpdef Ex Sin(Ex e):
#    return ex_to_Ex(sin(e._this))
cpdef Ex sqrt(x):
    return ex_to_Ex(_sqrt(py_to_ex(x)))
cpdef Ex abs(x):
    return ex_to_Ex(_abs(py_to_ex(x)))
cpdef Ex conjugate(x):
    return ex_to_Ex(_conjugate(py_to_ex(x)))
cpdef Ex real_part(x):
    return ex_to_Ex(_real_part(py_to_ex(x)))
cpdef Ex imag_part(x):
    return ex_to_Ex(_imag_part(py_to_ex(x)))
cpdef Ex step(x):
    return ex_to_Ex(_step(py_to_ex(x)))
cpdef Ex csgn(x):
    return ex_to_Ex(_csgn(py_to_ex(x)))
cpdef Ex eta(e1,e2):
    return ex_to_Ex(_eta(py_to_ex(e1),py_to_ex(e2)))
cpdef Ex exp(x):
    return ex_to_Ex(_exp(py_to_ex(x)))
cpdef Ex log(x): #natural polylogarithm
    return ex_to_Ex(_log(py_to_ex(x)))
cpdef Ex logb(e,b):
    return ex_to_Ex(_logb(py_to_ex(e),py_to_ex(b)))

cpdef Ex sin(x):
    return ex_to_Ex(_sin(py_to_ex(x)))
cpdef Ex cos(x):
    return ex_to_Ex(_cos(py_to_ex(x)))
cpdef Ex tan(x):
    return ex_to_Ex(_tan(py_to_ex(x)))
cpdef Ex sec(x):
    return ex_to_Ex(_sec(py_to_ex(x)))
cpdef Ex csc(x):
    return ex_to_Ex(_csc(py_to_ex(x)))
cpdef Ex cot(x):
    return ex_to_Ex(_cot(py_to_ex(x)))

cpdef Ex arcsin(x):
    return ex_to_Ex(_arcsin(py_to_ex(x)))
cpdef Ex arccos(x):
    return ex_to_Ex(_arccos(py_to_ex(x)))
cpdef Ex arctan(x):
    return ex_to_Ex(_arctan(py_to_ex(x)))
cpdef Ex arcsec(x):
    return ex_to_Ex(_arcsec(py_to_ex(x)))
cpdef Ex arccsc(x):
    return ex_to_Ex(_arccsc(py_to_ex(x)))
cpdef Ex arccot(x):
    return ex_to_Ex(_arccot(py_to_ex(x)))

cpdef Ex sinh(x):
    return ex_to_Ex(_sinh(py_to_ex(x)))
cpdef Ex cosh(x):
    return ex_to_Ex(_cosh(py_to_ex(x)))
cpdef Ex tanh(x):
    return ex_to_Ex(_tanh(py_to_ex(x)))
cpdef Ex sech(x):
    return ex_to_Ex(_sech(py_to_ex(x)))
cpdef Ex csch(x):
    return ex_to_Ex(_csch(py_to_ex(x)))
cpdef Ex coth(x):
    return ex_to_Ex(_coth(py_to_ex(x)))

cpdef Ex arcsinh(x):
    return ex_to_Ex(_arcsinh(py_to_ex(x)))
cpdef Ex arccosh(x):
    return ex_to_Ex(_arccosh(py_to_ex(x)))
cpdef Ex arctanh(x):
    return ex_to_Ex(_arctanh(py_to_ex(x)))
cpdef Ex arcsech(x):
    return ex_to_Ex(_arcsech(py_to_ex(x)))
cpdef Ex arccsch(x):
    return ex_to_Ex(_arccsch(py_to_ex(x)))
cpdef Ex arccoth(x):
    return ex_to_Ex(_arccoth(py_to_ex(x)))

cpdef Ex Li(m,x): #classical polylogarithm as well as multiple polylogarithm
    return ex_to_Ex(_Li(py_to_ex(m),py_to_ex(x)))
cpdef Ex Li2(x):
    return ex_to_Ex(_Li2(py_to_ex(x)))
cpdef Ex Li3(x):
    return ex_to_Ex(_Li3(py_to_ex(x)))
cpdef Ex zetaderiv(n,x): # Derivatives of Riemann's Zeta-function  zetaderiv(0,x)==zeta(x)
    return ex_to_Ex(_zetaderiv(py_to_ex(n),py_to_ex(x)))
cpdef Ex zeta(m,s=None):
    if s is None:
        return ex_to_Ex(_zeta(py_to_ex(m)))
    else:
        return ex_to_Ex(_zeta(py_to_ex(m),py_to_ex(s)))
cpdef Ex G(a,s,y=None): #multiple polylogarithm
    if y is None:
        return ex_to_Ex(_G(py_to_ex(a),py_to_ex(s)))
    else:
        return ex_to_Ex(_G(py_to_ex(a),py_to_ex(s),py_to_ex(y)))
cpdef Ex S(n,p,x): #Nielsen’s generalized polylogarithm
    return ex_to_Ex(_S(py_to_ex(n),py_to_ex(p),py_to_ex(x)))
cpdef Ex H(m,x): #harmonic polylogarithm
    return ex_to_Ex(_H(py_to_ex(m),py_to_ex(x)))
cpdef Ex gamma(n):
    return ex_to_Ex(_tgamma(py_to_ex(n)))
cpdef Ex lgamma(n):
    return ex_to_Ex(_lgamma(py_to_ex(n)))
cpdef Ex beta(m,n):
    return ex_to_Ex(_beta(py_to_ex(m),py_to_ex(n)))
cpdef Ex psi(n,x=None): #derivatives of psi function (polygamma functions)
    if x is None:
        return ex_to_Ex(_psi(py_to_ex(n)))
    else:
        return ex_to_Ex(_psi(py_to_ex(n),py_to_ex(x)))
cpdef Ex factorial(n):
    return ex_to_Ex(_factorial(py_to_ex(n)))
cpdef Ex binomial(n,k): #binomial coefficients
    return ex_to_Ex(_binomial(py_to_ex(n),py_to_ex(k)))
cpdef Ex Order(e): #order term function in truncated power series
    return ex_to_Ex(_Order(py_to_ex(e)))

cpdef Ex EllipticK(x): #Complete elliptic integral of the first kind.
    return ex_to_Ex(_EllipticK(py_to_ex(x)))
cpdef Ex EllipticE(x): #Complete elliptic integral of the second kind.
    return ex_to_Ex(_EllipticE(py_to_ex(x)))

cpdef Ex chebyshevT(n,x):
    return ex_to_Ex(_chebyshevT(py_to_ex(n),py_to_ex(x)))
cpdef Ex chebyshevU(n,x):
    return ex_to_Ex(_chebyshevU(py_to_ex(n),py_to_ex(x)))
cpdef Ex hermiteH(n,x):
    return ex_to_Ex(_hermiteH(py_to_ex(n),py_to_ex(x)))
cpdef Ex gegenbauerC(n,a,x):
    return ex_to_Ex(_gegenbauerC(py_to_ex(n),py_to_ex(a),py_to_ex(x)))

cpdef Ex sinIntegral(x):
    return ex_to_Ex(_sinIntegral(py_to_ex(x)))
cpdef Ex cosIntegral(x):
    return ex_to_Ex(_cosIntegral(py_to_ex(x)))
cpdef Ex sinhIntegral(x):
    return ex_to_Ex(_sinhIntegral(py_to_ex(x)))
cpdef Ex coshIntegral(x):
    return ex_to_Ex(_coshIntegral(py_to_ex(x)))
cpdef Ex logIntegral(x):
    return ex_to_Ex(_logIntegral(py_to_ex(x)))
cpdef Ex expIntegralEi(x):
    return ex_to_Ex(_expIntegralEi(py_to_ex(x)))
cpdef Ex expIntegralE(s, x):
    return ex_to_Ex(_expIntegralE(py_to_ex(s),py_to_ex(x)))
cpdef Ex legendreP(n,m,x):
    if m==0:
        return ex_to_Ex(_legendreP(py_to_ex(n),py_to_ex(x)))
    else:
        return ex_to_Ex(_legendreP(py_to_ex(n),py_to_ex(m),py_to_ex(x)))
cpdef Ex legendreQ(n,m,x):
    if m==0:
        return ex_to_Ex(_legendreQ(py_to_ex(n),py_to_ex(x)))
    else:
        return ex_to_Ex(_legendreQ(py_to_ex(n),py_to_ex(m),py_to_ex(x)))
cpdef Ex besselJ(n,x):
    return ex_to_Ex(_besselJ(py_to_ex(n),py_to_ex(x)))
cpdef Ex besselY(n,x):
    return ex_to_Ex(_besselY(py_to_ex(n),py_to_ex(x)))
cpdef Ex besselI(n,x):
    return ex_to_Ex(_besselI(py_to_ex(n),py_to_ex(x)))
cpdef Ex besselK(n,x):
    return ex_to_Ex(_besselK(py_to_ex(n),py_to_ex(x)))



cpdef lst lsolve(eqns, symbols, solve_algo opt=solve_algo_automatic):
    temequs=eqns
    temvars=symbols
    if not isinstance(eqns,list):
        temequs = [eqns]
    if not isinstance(symbols,list):
        temvars = [symbols]
    for i in range(len(temequs)):
        if not is_a[_relational](copy_Ex(temequs[i])._this):
            temequs[i] = relational(temequs[i],0)
    return ex_to_lst(_lsolve(_lst_to_ex(cpplist_to__lst(list_to_cpplist(temequs))),
    _lst_to_ex(cpplist_to__lst(list_to_cpplist(temvars))), opt))

cpdef Ex fsolve(Ex f, Ex x, x1, x2):
    if not is_a[_symbol](x._this):
        raise Exception("2nd argument should be a symbol.")
    elif not is_a[_numeric](py_to_ex(x1)) or not is_a[_numeric](py_to_ex(x2)):
        raise Exception("3rd and 4th, both arguments should be numeric.")
    return ex_to_Ex(_numeric_to_ex(_fsolve(f._this, ex_to[_symbol](x._this), ex_to[_numeric](py_to_ex(x1)),
    ex_to[_numeric](py_to_ex(x2)))))
cpdef Ex convert_H_to_Li(Ex parameterlst, Ex arg):
    return ex_to_Ex(_convert_H_to_Li(parameterlst._this,arg._this))


####################################################################################################
                                ###### function object ###############
####################################################################################################
cdef class function(Ex):
    pass

####################################################################################################
#                                                                                                  #
#                     ############# EXTENSIONS (NEW CLASSES) ###############                      #
#                                                                                                  #
####################################################################################################
cdef Ex create_infinity():
    return ex_to_Ex(_Infinity.eval())
Infinity=create_infinity()

####################################################################################################
                                ###### functions object ###############
####################################################################################################

cdef class functions(Ex):
    cdef _functions _thisFunctions
    def __init__(self,funcname=None,list fd=None, symbol_assumptions symboltype=complex):
        super().__init__()
        if funcname is not None:
            #print ex_to_Ex(cpplist_to_ex(list_to_cpplist(fd)))
            self._thisFunctions = _functions(funcname.encode("UTF-8"),list_to__lst(fd),symboltype)
            self._this = self._thisFunctions.to_ex()
    def total_diff(self,Ex var):
        return ex_to_Ex(self._thisFunctions.total_diff(var._this))

####################################################################################################
                                ###### Limit object ###############
####################################################################################################
cdef class Limit(Ex):
    cdef _Limit _thisLimit
    def __init__(self, e, Ex z, z0, str dir="+-"):
        super().__init__()
        self._thisLimit = _Limit(py_to_ex(e),z._this,py_to_ex(z0),dir.encode("UTF-8"))
        self._this = self._thisLimit.to_ex()

####################################################################################################
                                ###### Diff object ###############
####################################################################################################
cdef class Diff(Ex):
    cdef _Diff _thisDiff
    def __init__(self,Ex d=None, Ex i=None,int o=1):
        super().__init__()
        if d is not None:
            self._thisDiff = _Diff(d._this,i._this,ex(o))
            self._this = self._thisDiff.to_ex()

    def change_variable(self,Ex oldNewOrNewOld, Ex newvarName):
        return ex_to_Ex(self._thisDiff.changeVariable(oldNewOrNewOld._this,newvarName._this))

####################################################################################################
                                ###### Integrate object ###############
####################################################################################################
cdef class Integrate(Ex):
    cdef _Integrate _thisIntegrate
    def __init__(self,Ex integrand_=None, Ex var_=None, partial_num_l_= None, u_=None, int partial_num=-1):
        super().__init__()
        if integrand_ is not None:
            if u_ is None:
                if partial_num_l_ is None:
                    self._thisIntegrate = _Integrate(integrand_._this,var_._this,-1)
                else:
                    self._thisIntegrate = _Integrate(integrand_._this,var_._this,<int>(partial_num_l_))
            else:
                self._thisIntegrate = _Integrate(integrand_._this,var_._this,py_to_ex(partial_num_l_),py_to_ex(u_),partial_num)
            self._this=self._thisIntegrate.to_ex()

    def set_partial_num(self, int p):
        self._thisIntegrate.set_partial_num(p)

    def integrate(self, Ex var_, partial_num_l_= None, u_=None, int partial_num=-1):
        if u_ is None:
            if partial_num_l_ is None:
                return ex_to_Ex(_integrate(self._this, var_._this, -1))
            else:
                return ex_to_Ex(_integrate(self._this, var_._this, <int>(partial_num_l_)))
        else:
            return ex_to_Ex(_integrate(self._this,var_._this,py_to_ex(partial_num_l_),py_to_ex(u_),partial_num))

    def change_variable(self,Ex oldNewOrNewOld, Ex newvarName):
        return _Integrate_to_Integrate(self._thisIntegrate.changeVariable(oldNewOrNewOld._this,newvarName._this))

    def evaluate(self):
        return ex_to_Ex(self._thisIntegrate.evaluate())

cpdef Ex evaluate(Ex e):
    return ex_to_Ex(_evaluate(e._this))

cpdef Ex integrate(Ex expr_, Ex var_, partial_num_l_= None, u_=None, int partial_num=-1):
    if u_ is None:
        if partial_num_l_ is None:
            return ex_to_Ex(_integrate(expr_._this, var_._this, -1))
        else:
            return ex_to_Ex(_integrate(expr_._this, var_._this, <int>(partial_num_l_)))
    else:
        return ex_to_Ex(_integrate(expr_._this,var_._this,py_to_ex(partial_num_l_),py_to_ex(u_),partial_num))

cpdef Ex limit(e,Ex z, z0, str dir="+-"):
    return ex_to_Ex(_limit(py_to_ex(e),(z)._this, py_to_ex(z0), dir.encode("UTF-8")))

cpdef Ex collect_common_factor(Ex e):
    return ex_to_Ex(Collect_common_factor(e._this))

def Simplify(Ex expr, simplify_options rules=simplify_nooptions):
    if rules is simplify_nooptions:
        return ex_to_Ex(_fullSimplify(expr._this,_funcSimp))
    else:
        return ex_to_Ex(_fullSimplify(expr._this,rules))
cpdef Ex simplify(Ex expr, simplify_options rules=simplify_nooptions):
    if rules is simplify_nooptions:
        return ex_to_Ex(_simplify(expr._this,_funcSimp))
    else:
        return ex_to_Ex(_simplify(expr._this,rules))
cpdef Ex fullsimplify(Ex expr, simplify_options rules=simplify_nooptions):
    if rules is simplify_nooptions:
        return ex_to_Ex(_fullsimplify(expr._this,_funcSimp))
    else:
        return ex_to_Ex(_fullsimplify(expr._this,rules))

cpdef lst solve(equs_, vars_):
    temequs=equs_
    temvars=vars_
    if not isinstance(equs_,list):
        temequs = [equs_]
    if not isinstance(vars_,list):
        temvars = [vars_]
    return exsetlst_to_lst(_solve(cpplist_to__lst(list_to_cpplist(temequs)),
                               cpplist_to__lst(list_to_cpplist(temvars))))

#####conversion to numpy array##########
def to_numpy_array(e,list varList, list ndarrayList, is_meshgrid=False):
    import numpy
    if callable(e): #checking e is function
        if len(varList) is 1:
            return numpy.array([e(varList[0]) for varList[0] in ndarrayList[0]],dtype=numpy.float64)
        newNdarray = ndarrayList[0]
        for i in ndarrayList[1:]:
            if type(i[0]) is numpy.ndarray and type(newNdarray[0]) is numpy.ndarray:
                newNdarray=(numpy.concatenate((newNdarray,i)))
            elif type(newNdarray[0]) is numpy.ndarray:
                newNdarray=(numpy.concatenate((newNdarray,[i])))
            elif type(i[0]) is numpy.ndarray:
                newNdarray=(numpy.concatenate(([newNdarray],i)))
            else:
                newNdarray=(numpy.concatenate(([newNdarray],[i])))
        if(len(varList) is 2):
            return numpy.array([e(i,j) for (i,j) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 3):
            return numpy.array([e(i,j,k) for (i,j,k) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 4):
            return numpy.array([e(i,j,k,l) for (i,j,k,l) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 5):
            return numpy.array([e(i,j,k,l,m) for (i,j,k,l,m) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 6):
            return numpy.array([e(i,j,k,l,m,n) for (i,j,k,l,m,n) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        else:
            raise Exception("The maximum number of function argument is 6.")
    else:
        if len(varList) is 1:
            return numpy.array([(e.subs({varList[0]:i})).to_double() for i in ndarrayList[0]],dtype=numpy.float64)
        newNdarray = ndarrayList[0]
        for i in ndarrayList[1:]:
            if type(i[0]) is numpy.ndarray and type(newNdarray[0]) is numpy.ndarray:
                newNdarray=(numpy.concatenate((newNdarray,i)))
            elif type(newNdarray[0]) is numpy.ndarray:
                newNdarray=(numpy.concatenate((newNdarray,[i])))
            elif type(i[0]) is numpy.ndarray:
                newNdarray=(numpy.concatenate(([newNdarray],i)))
            else:
                newNdarray=(numpy.concatenate(([newNdarray],[i])))
        if(len(varList) is 2):
            return numpy.array([(e.subs({varList[0]:i,varList[1]:j})).to_double() for (i,j) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 3):
            return numpy.array([(e.subs({varList[0]:i,varList[1]:j,varList[2]:k})).to_double() for (i,j,k) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 4):
            return numpy.array([(e.subs({varList[0]:i,varList[1]:j,varList[2]:k,varList[3]:l})).to_double() for (i,j,k,l) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 5):
            return numpy.array([(e.subs({varList[0]:i,varList[1]:j,varList[2]:k,varList[3]:l,varList[4]:m})).to_double() for (i,j,k,l,m) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 6):
            return numpy.array([(e.subs({varList[0]:i,varList[1]:j,varList[2]:k,varList[3]:l,varList[4]:m,varList[5]:n})).to_double() for (i,j,k,l,m,n) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        elif(len(varList) is 7):
            return numpy.array([(e.subs({varList[0]:i,varList[1]:j,varList[2]:k,varList[3]:l,varList[4]:m,varList[5]:n,varList[6]:o})).to_double() for (i,j,k,l,m,n,o) in numpy.transpose(newNdarray)],dtype=numpy.float64)
        else:
            raise Exception("The maximum number of variables is 7.")


#### functions working on list ###########
#cpdef list expand(list inp):
#    return inp



#cdef class Add:
#    cdef add *_this
#    def _cinit_(self,Symbol s=None,Symbol l=None):
#        if s is not None and l is not None:
#            self._this=new add((s._this),(l._this))
#        else:
#            self._this=NULL

#cdef ex pysymboltoex(Symbol Sym):
#    cdef ex temex
#    temex=symboltoex(deref(Sym._thisptr))
#    return temex


