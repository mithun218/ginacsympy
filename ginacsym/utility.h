
/** @file utility.h
 *
 *  Interface to some usefull utilities implemented in utility.cpp file. */

/*
 *  Copyright (C) 2024 Mithun Bairagi <bairagirasulpur@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */


#ifndef UTILITY_H_INCLUDED
#define UTILITY_H_INCLUDED

#include <chrono>
#include <flint/flint.h>
#include "ginacwrapper.h"
#include "utils.h"

#include <stdio.h>  /* defines FILENAME_MAX */
//#ifdef WINDOWS
//    #include <direct.h>
//    #define GetCurrentDir _getcwd
//#else
//    #include <unistd.h>
//    #define GetCurrentDir getcwd
// #endif

namespace ginacsym{
//#define factor_all 18
#define Gtolerance ginacsym::pow(10,-10)
    //#define GiNaCDE_gui



    typedef std::map<ex, exvector, ex_is_less> exmapexvec;
    typedef std::set<lst,ex_is_less> exsetlst;

    /** numeric used in other files **/
    extern const numeric _numeric1,_numeric2,_numeric3,_numeric4;



    /** it measures the beginning time of evaluations **/
    extern std::chrono::time_point<std::chrono::system_clock> beginTime;


    //extern string CurrentPath, filename;

    /** The variables store solutions and constraints of NLPDEs **/
    extern std::vector<lst> solutionClt; extern lst constraints;

    /**It sets number of prrecision digits for flint arb,acb number**/
    extern slong _digits;

    bool has_only_digits(const std::string s);

    std::vector<std::string> split (const std::string &s, char delim);



    /** Symbol finder **/
    class symbol_finderc:public map_function
    {
    public:
        exset symbols;
        void clear()
        {
            symbols.clear();
        }
        ex expr_visitor(const ex& _expr);
        ~symbol_finderc(){}
    };

    extern symbol_finderc symbol_finder;
    exset symbols(const ex&);

    /** converting number into rational**/
    class dorat:public map_function
    {
    public:
        bool israt;
        dorat(){}
        void set(){israt = true;}
        ex expr_visitor(const ex& _e);
        ~dorat(){}
    };

    /** collecting power of each base from pow argument, excludes similar power**/
    class basepow_clt:public map_function
    {
    private:
        exset exprchk;
    public:
        std::map<ex, lst, ex_is_less> basepow;
        const ex _var;
        void clear()
        {
            basepow.clear();
            exprchk.clear();
        }
        basepow_clt(const ex _var_):_var(_var_){}
        ex expr_visitor(const ex& _e);
        ~basepow_clt(){}
    };

    /** Checking presence of functions with given variable in eqsn **/
    class funcpresent:public map_function
    {
    public:
        bool funcpresence = false,
            varInPow = false;
        exset func, funcwtvar;
        ex _var;
        funcpresent(ex _var_):_var(_var_){}
        ex expr_visitor(const ex& _e);
        ~funcpresent(){}
    };

    /** Calculating Factor of irrational function without expand. **/
    ex Factor(const ex&);

    /** calculating gcd of list of expressions **/
    ex Gcd(lst _exp);

    /** calculating gcd of list of expressions **/
    ex Lcm(lst _exp);

    /** replacing I by _symb **/
    class replaceI:public map_function
    {
    public:
        ex expr_visitor(const ex& _e);
        replaceI(){}
        ~replaceI(){}
    };

    /** Checking polynomial type in fim **/
    class polycheckFim:public map_function
    {
    public:
        bool polytype;
        ex _var;
        polycheckFim(bool _polytype, ex _var_):polytype(_polytype),_var(_var_){}
        ex expr_visitor(const ex& _e);
        ~polycheckFim(){}
    };

    /// Collecting powers of a variable in fim///
    class powClt:public map_function
    {
    public:
        exvector powers;
        ex _var;
        powClt(ex _var_):_var(_var_){}
        ex expr_visitor(const ex& _e);
        ~powClt(){}
    };


    /** Checking polynomial function **/

    bool is_poly(const ex& _expr, const ex& _var);


    /// doing conjugate free ///
    class conjuFree:public map_function
    {
    public:
        ex expr_visitor(const ex& _e);
        ~conjuFree(){}
    };

    extern conjuFree conjuFreee;

    /** collecting coefficients of variables in base of non-integer power. **/

    inline ex Collect(const ex& _expr, const ex& _var) // This function has been used in odeType_check function
    {                                                  // in desolve.cpp file.
        ex temexpr = _expr;

        const ex C1_=symbol("C1_");

        if(is_a<power>(_expr) && _expr.op(0).has(_var) && _expr.op(0).is_polynomial(_var)&& _expr.op(1)==_ex1_2)
            temexpr = pow(collect(C1_*expand(_expr.op(0)), _var),_expr.op(1));
        else if(_expr.is_polynomial(_var))
            temexpr = collect(C1_*expand(_expr), _var);

        return temexpr;
    }

    /** replacing power having "add" base with generated symbols **/ // used in Factor
    class powBaseSubs:public map_function
    {
        unsigned j;
        ex expr;
        std::string str;
    public:
        bool isNu;
        size_t addNum;
        exmap exprToSymMap;
        powBaseSubs( unsigned j_): j(j_){exprToSymMap.clear();addNum=0;isNu=false;}
        ex expr_visitor(const ex& _e);
        ~powBaseSubs(){}
    };

    /** this function substitute generated symbols from exmap **/
    ex genSymbSubs(const ex& _e, const exmap& highDegSubsClt);

    /** Getting lst of coefficients from all terms where present _var.
isCltPowZero = true allow to get coefficients of _var^0. **/
    lst collectAllCoeff(const ex& _expr, const lst& _var, const bool& isCltPowZero, exmap& _varsWtcoeff);

    /** Getting numerator and denominator.
 *  This functution determines numer/denom
 *  accurately having fractional power.
 *  Such as: Numer_Denom(1/(x/y)^(1/2)) returns {1,(x/y)^(1/2)}, but, cuurently, Ginac,s
 *  numer_denom does not give correct results.
 *  To avoid wrong results, this function replace all the base with fractional power by generated symbol, then it
 *  determines numer/denom.
 *  **/
    ex Numer_Denom(const ex& _expr);

    /**Following function checks presence of functions (Ex: sin,cos,log,asin etc.)**/
    bool hasFunction(const ex& _expr);

   /**collect all coefficients of x in polynomial or nonpolynomial expressions**/
    class collectAllc:public map_function{
    public:
        bool distributed;
        ex var;
        exmap ma;
        collectAllc(const ex& _var,bool _distributed):distributed(_distributed),var(_var){}
        ex expr_visitor(const ex& e);
    };

    inline ex collectAll(const ex& e,const ex& _var,bool _distributed=false)
    {
        collectAllc collectAlld(_var,_distributed);
        return collectAlld.expr_visitor(e);
    }

    /** Finds all symbols in an expression. Used by factor_sqrfree(), factor(),quo().*/
    class find_symbols_map : public map_function
    {
    public:
        exset syms;
        ex expr_visitor(const ex& e) override
        {
            if ( is_a<symbol>(e) ) {
                syms.insert(e);
                return e;
            }
            return e.map(*this);
        }
    };

    /** getting all symbols from expression **/
    inline exset get_symbols(const ex& e)
    {
        find_symbols_map find_symbols_mapd;
        find_symbols_mapd.expr_visitor(e);
        return find_symbols_mapd.syms;
    }
    /**checking whether it is number**/
    inline bool is_number(const ex& e)
    {
        for(const_preorder_iterator itr=e.preorder_begin();itr!=e.preorder_end();itr++){
            if(is_a<symbol>(*itr))
                return false;
        }
        return true;
    }

    /**replacing nonrational terms with generated symbols**/
    class nonRatsubs: public map_function
    {
    public:
        ex var,newSym;
        exmap syms;
        unsigned symnum=0;
        nonRatsubs(const ex& e):var(e){}

        ex expr_visitor(const ex& e)
        {
            if(!e.has(var) && !e.info(info_flags::rational)){
                newSym=generator.symGenerator("ratSymb_" + to_string(symnum));
                syms[newSym]=e;
                symnum=symnum+1;
                return newSym.map(*this);
            }
            else return e.map(*this);
        }

    };

    inline void _set_digits(const long& prec){
        Digits=prec;//set precesion for cln
        _digits=prec*3.33;//set precesion for flint
    }

    inline long _get_digits(){
        return Digits;
    }


    //A slightly different implementation of map() that allows applying algebraic functions
    //to operands. The second argument to map() is an expression containing the wildcard ‘$0’ which
    //acts as the placeholder for the operands (it works like ginac's ginsh map):
    class apply_map_function : public map_function {
        ex apply;
    public:
        apply_map_function(const ex & a) : apply(a) {}
        virtual ~apply_map_function() {}
        ex expr_visitor(const ex & e) override { return apply.subs(wild() == e, true); }
    };

    static ex f_map(const ex &e1,const ex &e2)
    {
        apply_map_function fcn(e2);
        return e1.map(fcn);
    }
}

#endif // UTILITY_H_INCLUDED
