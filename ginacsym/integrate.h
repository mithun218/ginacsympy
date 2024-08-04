
/** @file integrate.h
 *
 *  Interface to the class integratec and function integrate implemented in integrate.cpp. */

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

#ifndef INTEGRATE_H_INCLUDED
#define INTEGRATE_H_INCLUDED


#include "inert.h"
#include "matrix.h"
#include "utils.h"
#include "relational.h"
#include "operators.h"
#include "lst.h"

namespace ginacsym {

    class integratec
    {
        const ex var, IntDiffargu;
        int partial_num_count=0;//set number of step of partial integration
//        ex partialIntegrandU,partialIntegrandV;
//        bool isFirstTime=true;
        ex find_const(const ex& expr_) const;
        ex do_inte(const ex& expr_) const;
        ex substituttion(const exmap& newvari);
    public:
        int partial_num;//if partial_num<0, partial integration is performed until it is terminated.
        integratec(const ex& var_, const int partial_num=-1):var(var_),partial_num(partial_num){}
        ex partial_integration(const ex& expr_, const bool& isFirstTime=true, const ex& partialIntegrandU=_ex0,
                               const ex& partialIntegrandV=_ex0);
        ex operator()(const ex& expr_);

        ~integratec(){}
    };

    ///** integrating functions **////
    /// \brief integrate
    /// \param expr_ integrand
    /// \param var_ variable
    /// \param partial_num number of step in partial integration. Default is -1 and partial integration
    /// is performed until it is terminated. But, at present this terminating process is not guranted, so, one can
    /// stop the partial integration using this parameter.
    ///
inline ex integrate(const ex& expr_, const ex& var_, const int& partial_num=-1)
{
    integratec integratef(var_,partial_num);
    if (is_a<matrix>(expr_)) {
        matrix exprm=ex_to<matrix>(expr_);
        const unsigned r = exprm.rows(),c=exprm.cols();
        for (unsigned i = 0; i < r; ++i) {
            for (unsigned j = 0; j < c; ++j) {
                exprm(i,j)=integratef(exprm(i,j));
            }
        }
        return exprm;
    }
    else if (is_a<lst>(expr_)) {
        lst exprl=ex_to<lst>(expr_);
        for (unsigned i = 0; i < nops(exprl); ++i) {
            exprl.let_op(i)=integratef(exprl.op(i));
        }
        return exprl;
    }
    else return integratef(expr_);
}
inline ex integrate(const ex& expr_, const ex& var_,const ex& l_, const ex& u_, const int& partial_num=-1)
{
    integratec integratef(var_,partial_num);

    if (is_a<matrix>(expr_)) {
        matrix exprm=ex_to<matrix>(expr_);
        const unsigned r = exprm.rows(),c=exprm.cols();
        for (unsigned i = 0; i < r; ++i) {
            for (unsigned j = 0; j < c; ++j) {
                const ex intevalue=integratef(exprm(i,j));
                if(is_a<Integrate>(intevalue))
                    exprm(i,j) = Integrate(intevalue.op(0),var_,l_,u_);
                else exprm(i,j) = intevalue.subs(var_==u_)-intevalue.subs(var_==l_);
            }
        }
        return exprm;
    }
    else if (is_a<lst>(expr_)) {
        lst exprl=ex_to<lst>(expr_);
        for (unsigned i = 0; i < nops(exprl); ++i) {
            const ex intevalue=integratef(exprl.op(i));
            if(is_a<Integrate>(intevalue))
                exprl.let_op(i) = Integrate(intevalue.op(0),var_,l_,u_);
            else exprl.let_op(i) = intevalue.subs(var_==u_)-intevalue.subs(var_==l_);
        }
        return exprl;
    }
    else{
        const ex intevalue=integratef(expr_);
        if(is_a<Integrate>(intevalue))
            return Integrate(intevalue.op(0),var_,l_,u_);
        else return intevalue.subs(var_==u_)-intevalue.subs(var_==l_);
    }
}

exvector Liate(const ex& expr_, const ex &var);
ex func_inte(const ex& expr_, const ex& var, int partial_num);
}

#endif // INTEGRATE_H_INCLUDED
