#ifndef FUNCTIONS_H
#define FUNCTIONS_H
////#include "ex.h"
//#include "inifcns2.h"
////#include "lst.h"
//#include "ginacwrapper.h"
//#include "ginac.h"
#include "ginacwrapper.h"
#include "inert.h"
#include "inifcns.h"
#include "basic.h"
#include "ex.h"
#include "symbol.h"
#include "operators.h"
#include "relational.h"

namespace ginacsym{

class functions:public basic {
    GINAC_DECLARE_REGISTERED_CLASS(functions, basic)

    ex functionname;
    bool islatexname;
    std::string fns;
    lst functiondependency = {};
    unsigned assumption;
public:

  functions(const std::string &fns, const lst &fd, unsigned assu);
 protected:
   void do_print(const print_context & c, unsigned level) const{
       functionname.print(c);
   }
   void do_print_latex(const print_latex & c, unsigned level) const{
       if(islatexname){
            c.s<<"\\";c.s<<fns.substr(1,fns.npos-1);
            c.s<<"\\left(";
            for(size_t i=0;i<(functiondependency).nops();i++){
               if(i!=0)
               c.s<<",";
               functiondependency[i].print(c);
            }
           c.s<<"\\right)";
       }
       else{
           c.s<<fns;
           c.s<<"\\left(";
           for(size_t i=0;i<(functiondependency).nops();i++){
                if(i!=0)
                c.s<<",";
                functiondependency[i].print(c);
           }
           c.s<<"\\right)";
       }

   }
 public:
    ex to_ex() const{return *this;}

    bool has(const ex& e, unsigned int opt=0) const override{
        return functiondependency.has(e,opt);
    }
    ex total_diff(const ex& s) const{
        if(functiondependency.has(s) and functiondependency.nops()>1){

            ex tem=Diff((*this),s,1);
            //chain rule
            for(size_t i=0;i<functiondependency.nops();i++){
                if(!functiondependency[i].is_equal(s)){
                    if(is_a<functions>(functiondependency[i]))
                        tem=tem+ Diff((*this),functiondependency[i],1)*(ex_to<functions>(functiondependency[i]).total_diff(s));
                    else
                        tem=tem+ Diff((*this),functiondependency[i],1)*Diff(functiondependency[i],s);
                }
            }
            return tem;
        }
        else if (functiondependency.has(s))
            return Diff((*this),s,1);
        else return 0;
    }
    ex derivative(const symbol& s) const override{
        if (functiondependency.has(s))
            return Diff((*this),s,1);
        else return 0;
    }

    size_t nops() const override{
        return functiondependency.nops();
    }

    ex op(size_t i) const override{
       if(i<functiondependency.nops())
           return functiondependency[i];
       else
           throw std::range_error("sprod::op(): no such operand");
    }
//    ex& let_op(size_t i) override;

    ex subs(const exmap& m, unsigned options = 0) const override;
    ex map(map_function & f) const override { return (*this); }
    ex expand(unsigned options = 0) const override{return (*this);}

    bool info(unsigned inf) const override;

   ex imag_part() const override{
       if(assumption==symbol_assumptions::possymbol || assumption==symbol_assumptions::realsymbol) return 0;
       else return imag_part_function(*this).hold();
   }
   ex real_part() const override{
       if(assumption==symbol_assumptions::possymbol || assumption==symbol_assumptions::realsymbol) return (*this);
       else return (real_part_function(*this).hold());
   }
   ex conjugate() const override{
       if(assumption==symbol_assumptions::possymbol || assumption==symbol_assumptions::realsymbol) return (*this);
       else return conjugate_function(*this).hold();
   }

};
//GINAC_DECLARE_UNARCHIVER(functions);


}



#endif // FUNCTIONS_H
