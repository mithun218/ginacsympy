
/** @file outform.h
 *
 *  Interface to the GiNaCDE's output formt implemented in outform.cpp. */


#ifndef OUTFORM_H_INCLUDED
#define OUTFORM_H_INCLUDED

//#include "ex.h"
#include <string>

namespace ginacsym{

    #define maple 8
    #define mathematica 9
    #define ginac 10

    extern int output;
    extern std::string dpndtWtIndpndt; // It is assinged to dependend variable with independen variable(s), such as u(t,x)

    std::string outstr(const char* _sym, int symno);

    std::string replacestring(std::string subject, const std::string& search,
                              const std::string& replace);

    bool bktmch(const std::string _instr);

//    std::string to_numpy_string(const ex& expr, const std::string& prefix="numpy");

    //string gmathematica(string _instr);

    //string diffformchange(const ex& diffeq, const lst& dpndt_var, const exset& indpndt_var);

    //string writetofile(stringstream&, const ex& dpndt_var);
}

#endif // OUTFORM_H_INCLUDED
