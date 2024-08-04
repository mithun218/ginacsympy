/** @file inert.h
 *
 *  Implementation of class of Diff, Integrate. **/
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

#ifndef INERT_H
#define INERT_H

#include "basic.h"
#include "ex.h"

namespace ginacsym {

class Diff: public basic
{
    GINAC_DECLARE_REGISTERED_CLASS(Diff,basic)
public:
    Diff(const ex& d, const ex& i, const ex& o=1);
protected:
    void do_print(const  print_context& c,unsigned level) const;
    void do_print_python(const  print_python& c,unsigned level) const;
    void do_print_latex(const  print_latex& c,unsigned level) const;
    ex derivative(const symbol& s) const override;
public:
    ex to_ex() const{return *this;}
    ex eval() const override;
    ex eval_ncmul(const exvector & v) const override;
    size_t nops() const override {return 3;}
    ex op(const size_t i) const override;
    ex& let_op(size_t i) override;
    ex subs(const exmap& m, unsigned options = 0) const override;
    bool has(const ex& e, unsigned opt=0) const override{return de.has(e,opt);}
    ex expand(unsigned opt=0) const override;
   //exvector get_free_indices() const override;
    unsigned return_type() const override;
    return_type_t return_type_tinfo() const override;

           /** change of variable **/
    //old variable to new variable or
    //new variable to old variable
    ex changeVariable(const ex& oldNewOrNewOld, const ex& newvarName) const;
    ex evaluate()  const;

private:
    ex de,ind,order;

};
bool is_partial(const ex& expr);


class Integrate:public basic
{
    GINAC_DECLARE_REGISTERED_CLASS(Integrate,basic)

protected:
    void do_print(const print_context & c, unsigned level) const;
    void do_print_python(const print_python & c, unsigned level) const;
    void do_print_latex(const print_latex & c, unsigned level) const;
    ex derivative(const symbol & s) const override;
//    ex series(const relational & r, int order, unsigned options = 0) const override;

public:
    Integrate(const ex& integrand_, const ex& var_, const int& partial_num_=-1):
        integrand(integrand_),var(var_),partial_num(partial_num_),is_definite(false){}
    Integrate(const ex& integrand_, const ex& var_,const ex& l_, const ex& u_, const int& partial_num_=-1):
        integrand(integrand_),var(var_),l(l_),u(u_),partial_num(partial_num_),is_definite(true){}

    ex to_ex() const{return *this;}
    // functions overriding virtual functions from base classes
    unsigned precedence() const override {return 45;}
    ex eval() const override;
//    ex evalf() const override;
//    int degree(const ex & s) const override;
//    int ldegree(const ex & s) const override;
    ex eval_ncmul(const exvector & v) const override;
    size_t nops() const override;
    ex op(size_t i) const override;
    ex & let_op(size_t i) override;
    ex subs(const exmap& m, unsigned options = 0) const override;
    bool has(const ex& e, unsigned opt=0) const override{return integrand.has(e,opt);}
    ex expand(unsigned options = 0) const override;
    //exvector get_free_indices() const override;
    unsigned return_type() const override;
    return_type_t return_type_tinfo() const override;
    ex conjugate() const override;
    ex integrate(const ex& var_, const int& partial_num_=-1) const
    {
        return dynallocate<Integrate>(Integrate(*this,var_,partial_num_));
    }

    ex integrate(const ex& var_,const ex& l_, const ex& u_, const int& partial_num_=-1) const
    {
        return dynallocate<Integrate>(Integrate(*this,var_,l_, u_,partial_num_));
    }

    void set_partial_num(const int& p)
    {
        partial_num=p;
    }
        /** change of variable **/
    Integrate changeVariable(const ex& oldNewOrNewOld, const ex& newvarName) const;//old variable to new variable or
    ex apply_partial_integration() const;                                                                              //new variable to old variable
    ex evaluate()  const;

private:
    ex integrand,var,l,u;
    int partial_num;
    bool is_definite;
};

class Limit: public basic
{
    GINAC_DECLARE_REGISTERED_CLASS(Limit,basic)
public:
    /***
    //    Parameters
    //    ==========

    //    e : expression, the limit of which is to be taken

    //    z : symbol representing the variable in the limit.
    //        Other symbols are treated as constants. Multivariate limits
    //        are not supported.

    //    z0 : the value toward which ``z`` tends. Can be any expression,
    //        including ``oo`` and ``-oo``.

    //    dir : string, optional (default: "+-")
    //        The limit is bi-directional if ``dir="+-"``, from the right
    //        (z->z0+) if ``dir="+"``, and from the left (z->z0-) if
    //        ``dir="-"``. For infinite ``z0`` (``oo`` or ``-oo``), the ``dir``
    //        argument is determined from the direction of the infinity
    //        (i.e., ``dir="-"`` for ``oo``).
    ***/
    Limit(const ex& e,const ex& z,const ex& z0, const std::string& dir);
protected:
    void do_print(const  print_context& c,unsigned level) const;
    void do_print_python(const  print_python& c,unsigned level) const;
    void do_print_latex(const  print_latex& c,unsigned level) const;
//    ex derivative(const symbol& s) const override;
public:
    ex to_ex() const{return *this;}
    std::string get_direction()const {return dir;};
//    ex eval() const override;
    ex eval_ncmul(const exvector & v) const override;
    size_t nops() const override {return 3;}
    ex op(const size_t i) const override;
    ex& let_op(size_t i) override;
    ex subs(const exmap& m, unsigned options = 0) const override;
    bool has(const ex& e, unsigned opt=0) const override{return e.has(expr,opt);}
    ex expand(unsigned opt=0) const override;
    //exvector get_free_indices() const override;
    unsigned return_type() const override;
    return_type_t return_type_tinfo() const override;
    ex evaluate()  const;

private:
    ex expr,var,value;
    std::string dir;

};

class apply_partial_integration_on_exc: public map_function
{
public:
    apply_partial_integration_on_exc() {}
    ex expr_visitor(const ex& e);
};

class evaluatec: public map_function
{
public:
    evaluatec() {}
    ex expr_visitor(const ex& e);
};

ex apply_partial_integration_on_ex(const ex& e, const unsigned& partial_num);
ex evaluate(const ex& e);
}


#endif // INERT_H
