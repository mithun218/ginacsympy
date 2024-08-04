
/** @file solve.h
 *
 *  Interface to solve function implemented in solve.cpp. */


#ifndef SOLVE_H_INCLUDED
#define SOLVE_H_INCLUDED

#include "utility.h"

using namespace std;

namespace ginacsym{
/** counting total number of add in an equation**/
class totalAddInEq:public map_function
{

public:
    size_t addNum;
    totalAddInEq(){}
    ex expr_visitor(const ex& _e);
    ~totalAddInEq(){}
};


class solvec
{
    lst soluClt;
    lst one_eq_solu;
    exsetlst solu;
    exset SysEquCltEnv;
    totalAddInEq totalAddInEqV;

    /// for single polynomial solve ////////
    bool isexpanded=false;
    unsigned int symNum=1;
    lst soluCltPoly={};
    ex factortermprev=_ex1,lowestfactorterm;

    lst quadratic(const ex & equ_, const ex& var_);
    ex sqroot(const ex& _exp);

    int one_eq_solutions(const ex& _equ, const ex& _var);
    ex sysequ_red(const exset& sysequ_, const exset& var_);
    bool isVarPrsnt(const ex& _expr, const exset& _var);
    exsetlst solu_subs(set<lst, ex_is_less> solu_);
    lst varOrderByDeg(const lst& low_var, map<unsigned, ex>& eqDivider, unsigned& eqDividerSz);
//    lst polySoluWtAutoDegSelct(const ex& _equ, const ex& _var);

public:
    lst polySolve(const ex& _equ, const ex& _var, unsigned tryno=1);
    lst cubics(const ex & equ_, const ex& var_);
    lst Ncubics(const ex & equ_, const ex& var_);
    lst Nquartic(const ex & equ_, const ex& var_);
    exsetlst operator()(const lst & equ_, const lst& var_);
    ~solvec(){}
};

/** replacing the "pow" terms with _var with created symbols **/
class powBaseSubsWtVar:public map_function
{
    unsigned j;
    ex expr, _var;
    string str;
public:
    exmap exprToSymMapWtVar;
    powBaseSubsWtVar( unsigned j_,ex _var_): j(j_),_var(_var_){exprToSymMapWtVar.clear();}
    ex expr_visitor(const ex& _e);
    ~powBaseSubsWtVar(){}
};

/** replacing the "pow" terms without _var with created symbols **/
class powBaseSubsWtoutVar:public map_function
{
    unsigned j;
    ex expr, _var;
    string str;
public:
    exmap exprToSymMapWtoutVar;
    powBaseSubsWtoutVar( unsigned j_,ex _var_): j(j_),_var(_var_){exprToSymMapWtoutVar.clear();}
    ex expr_visitor(const ex& _e);
    ~powBaseSubsWtoutVar(){}
};

/** solve nonlinear equations **/
inline exsetlst solve(const lst& equs_, const lst& vars_)
{
    solvec solvef;
    return solvef(equs_, vars_);
}
}
#endif // SOLVE_H_INCLUDED
