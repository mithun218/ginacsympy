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



#ifndef GINACWRAPPER_H
#define GINACWRAPPER_H

#include "basic.h"
#include "clifford.h"
#include "color.h"
#include "ex.h"
#include "fail.h"
#include "matrix.h"
#include "ncmul.h"
#include "operators.h"
#include "indexed.h"
#include "sstream"

#include "numeric.h"
#include "power.h"
#include "relational.h"
#include "pseries.h"

// #ifndef IN_GINACSYM
#include "parser.h"
// #else
// #include "parser/parser.h"
// #endif


//error class
namespace ginacsym {
class unsupported_symbol{};
class unexpected_error{};
}

namespace ginacsym
{
extern  std::ostringstream strstr,latexstr,pythonstr;

//all generated symbols,functions are stored here and ex are also generated here.
class generatorc
{
    std::map<std::string, ex> symboldirectory;
    std::map<std::string,unsigned> symboltypes;
    parser reader;
public:
    generatorc() {}
    ex& symGenerator(const std::string& s, unsigned symboltype=symbol_assumptions::symbol, const bool& islatexname=false);
    ex exGenerator(const std::string& s, unsigned symboltype=symbol_assumptions::symbol, const bool& islatexname=false);
    ex functionSymbolFromString(const std::string& s, unsigned symboltype=symbol_assumptions::symbol);

    int symRegister(const ex& syms);
    ex exGeneratorFromString(const std::string& s) const;
    std::map<ex,unsigned,ex_is_less> allinfo() const;
    std::map<ex, unsigned, ex_is_less> aninfo(const ex& e) const;
};

extern generatorc generator;

/** this class genarates symbols used in other files **/
class externSymbolsc{
public:
    ex symb_,factSymb_, nan;
    externSymbolsc();
};
extern externSymbolsc externSymbols;

inline std::string to_string(const ex& expr)
{
    strstr.str("");
    strstr<<expr;
    return strstr.str();
}

inline std::string to_latex_string(const ex& expr)
{
    if(is_a<lst>(expr)){
        latexstr<<latex;
        latexstr.str("");
        latexstr<<ex_to<lst>(expr);
        return latexstr.str();
    }
    else{
        latexstr<<latex;
        latexstr.str("");
        latexstr<<expr;
        std::string tem=latexstr.str();

        if(!is_a<matrix>(expr)){
            std::string
                replace_of1="\\left\\left(",replace_of2="\\right\\right)",
                replace_by1="\\left(",replace_by2="\\right)";
            size_t pos=tem.find('(');
            while (pos!=std::string::npos) {
                tem.replace(pos,sizeof('('),replace_by1);
                pos=tem.find('(',pos+replace_by1.size());
            }
            pos=tem.find(replace_of1);
            while (pos!=std::string::npos) {
                tem.replace(pos,replace_of1.size(),replace_by1);
                pos=tem.find(replace_of1,pos+replace_by1.size());
            }

            pos=tem.find(')');
            while (pos!=std::string::npos) {
                tem.replace(pos,sizeof(')'),replace_by2);
                pos=tem.find(')',pos+replace_by2.size());
            }
            pos=tem.find(replace_of2);
            while (pos!=std::string::npos) {
                tem.replace(pos,replace_of2.size(),replace_by2);
                pos=tem.find(replace_of2,pos+replace_by2.size());
            }
        }

        return tem;
    }
}

inline std::string to_python_string(const ex& expr)
{
    strstr.str("");
    strstr<<expr;
    std::string tem = strstr.str();

    size_t pos=tem.find('^');
    while (pos!=std::string::npos) {
        tem.replace(pos,1,"**");
        pos=tem.find('^',pos+2);
    }
    return tem;
}



inline ex _numeric_to_ex(const numeric& n){
    return n;
}

inline ex _matrix_to_ex(const matrix& m)
{
    return m;
}

inline ex _relational_to_ex(const relational& r)
{
    return r;
}

inline ex _pseries_to_ex(const pseries& r)
{
    return r;
}

inline ex _clifford_to_ex(const clifford& r)
{
    return r;
}

inline ex _minkmetric_to_ex(const minkmetric& r)
{
    return r;
}

inline indexed _clifford_to__indexed(const clifford& r)
{
    return r;
}

inline ex _color_to_ex(const color& r)
{
    return r;
}

inline indexed _color_to__indexed(const color& r)
{
    return r;
}

inline ex _ncmul_to_ex(const ncmul& r)
{
    return r;
}

inline numeric string_to__numeric(const std::string& inp)
{
    return numeric(inp.c_str());
}
//inline bool _find(const ex& expr,const ex& pattern, std::vector<ex>& v)
//{
//    exset tem;
//    const bool ret=expr.find(pattern,tem);
//    std::copy(tem.begin(),tem.end(),std::back_inserter(v));
//    return ret;
//}

inline bool is_lst(const ex& l)
{
    return is_a<lst>(l);
}

inline void prepend(ex & a,const ex & b)
{
    lst tem= ex_to<lst>(a);
    a=tem.prepend(b);
}
inline void append(ex & a,const ex & b)
{
    lst tem= ex_to<lst>(a);
    a=tem.append(b);
}
inline void remove_first(ex & a)
{
    lst tem= ex_to<lst>(a);
    a=tem.remove_first();
}
inline void remove_last(ex & a)
{
    lst tem= ex_to<lst>(a);
    a=tem.remove_last();
}
inline void remove_all(ex & a)
{
    lst tem= ex_to<lst>(a);
    a=tem.remove_all();
}
inline void sort(ex & a)
{
    lst tem= ex_to<lst>(a);
    a=tem.sort();
}
inline void unique(ex & a)
{
    lst tem= ex_to<lst>(a);
    a=tem.unique();
}

inline matrix _Matrix(const std::list<std::list<ex>>& inpma)
{

    lst tem={};
    unsigned r=0, c=0;
    for(auto itr:inpma){
        r++;c=0;
        for(auto itr1:itr){
            tem.append(itr1);
            c++;
        }
    }
    return matrix(r,c,tem);
}

//spinidx spinidx_toggle_variance(const spinidx& inp)
//{
//    return inp.toggle_variance();
//}

}

#endif // GINACWRAPPER_H
