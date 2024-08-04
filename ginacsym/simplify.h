
/** @file simplify.h
 *
 *  Interface to GiNaCDE's simplify function implemented in simplify.cpp. */


#ifndef SIMPLIFY_H_INCLUDED
#define SIMPLIFY_H_INCLUDED


#include "numeric.h"
#include "operators.h"
#include "relational.h"


namespace ginacsym{

    extern size_t expandLevel, addNumFrFactr;// these variables limit the simplification of algebraic expressions
    extern numeric largstNumsimp; // this is the maximum number for simplification

    class simplifyc
    {
        //int rules = AlgSimp;
        exmap   AlgSimpRules1,AlgSimpRules2,
                TrigSimpRules1, TrigSimpRules2,TrigCombineRules,
                HyperSimpRules1, HyperSimpRules2,HyperCombineRules,
                logSimpRules,
                JacobiSimpRules1, JacobiSimpRules2;
        int SetRules(const unsigned int m = simplify_options::AlgSimp);
    public:
        simplifyc(){}
        ex algSimp(const ex& e);
        ex operator()(const ex& e, const unsigned int& rules = simplify_options::AlgSimp);
        ~simplifyc(){}
    };

    /**This collect common numerical factors inside base of fractional power.**/
    class Collect_common_factorsc:public map_function
    {
        ex temex, temex2, temex3;
    public:
        Collect_common_factorsc(){}
        ex expr_visitor(const ex& _e);
        ~Collect_common_factorsc(){}
    };


    class TrigArgSign_Complx:public map_function
    {
        ex var_;
    public:
        ex expr_visitor(const ex & e);
        ~TrigArgSign_Complx(){}
    };


    /** expanding terms containing inverse power **/
    class expandinv:public map_function
    {
        exmap repls;
    public:
        ex expr_visitor(const ex& e);
        ~expandinv(){}
    };

    /**doing factors in power argument **/
    class arguSimplify:public map_function
    {
    public:
        ex expr_visitor(const ex& e);
        ~arguSimplify(){}
    };



    /** doing number simplify **/
    class numSimplify:public map_function
    {
    public:
        std::map<numeric,numeric> primefactrs;
        ex getPrimefactors(const ex &e, const ex &fractimes);
        ex expr_visitor(const ex& e);
        ~numSimplify(){}
    };

    /** basic simplification function to apply algebraic rules **/
    //ex Simplify(const ex& _e, const int& rules = AlgSimp);

    /** replacing the "pow" terms with created symbols, which have less degree than expandLevel and base is in add container. **/
    class powBaseSubsLessThanDegLvl_1:public map_function
    {
        unsigned j;
        ex expr;
        std::string str;
    public:
        bool isNu;
        int addNum;
        exmap exprToSymMap;
        powBaseSubsLessThanDegLvl_1(){}
        void set(void)
        {
            j = 0;
            exprToSymMap.clear();addNum=0;isNu=false;
        }
        ex expr_visitor(const ex& _e);
        ~powBaseSubsLessThanDegLvl_1(){}
    };

    /** replacing the "pow" terms with created symbols, which have degree less than expandLevel and base is in add container. **/
    class fracPowBasSubsLvl_1:public map_function
    {
        unsigned j;
        ex expr, tem;
        ex numer_denomClt;
        std::string str;
    public:
        exmap baseCltLvl_1;
        fracPowBasSubsLvl_1(){}
        void set(void)
        {
            j = 0;
            baseCltLvl_1.clear();
        }
        ex expr_visitor(const ex& e);
        ~fracPowBasSubsLvl_1(){}
    };

    /** replacing the "pow" terms with created symbols, which have less degree than expandLevel and base is in add container. **/
    class fracPowBasSubsFactor:public map_function
    {
        unsigned j;
        ex expr, tem;
        ex numer_denomClt;
        std::string str;
    public:
        exmap baseClt;
        fracPowBasSubsFactor(){}
        void set(void)
        {
            j = 0;
            baseClt.clear();
        }
        ex expr_visitor(const ex& e);
        ~fracPowBasSubsFactor(){}
    };


    /** replacing the "pow" terms with created symbols, which have less degree than expandLevel and base is in add container. **/
    class powBaseSubsLessThanDeg:public map_function
    {
        unsigned j;
        ex expr, tem;
        ex numer_denomClt;
        std::string str;
        powBaseSubsLessThanDegLvl_1 Lvl_1;
    public:
        int addNum;
        exmap exprToSymMap;
        powBaseSubsLessThanDeg( unsigned j_): j(j_)
        {
            exprToSymMap.clear();
            addNum=0;
            Lvl_1.set();
        }
        ex expr_visitor(const ex& _e);
        ~powBaseSubsLessThanDeg(){}
    };


    /** replacing base of fractional power with generated symbols  **/
    class fracPowBasSubs:public map_function
    {
        unsigned j;
        ex expr, tem;
        ex numer_denomClt;
        std::string str;
        fracPowBasSubsLvl_1 Lvl_1;
    public:
        exmap baseClt;
        fracPowBasSubs(){}
        void set(void)
        {
            j = 0;
            Lvl_1.set();
            baseClt.clear();
        }
        ex expr_visitor(const ex& e);
        ~fracPowBasSubs(){}
    };

    /** replacing some functions with generated symbols  **/
    class funcSubs:public map_function
    {
        unsigned j;
        ex expr,expr2,expr3;
        std::string str;
    public:
        exmap baseClt;
        funcSubs(){j=0;baseClt.clear();}
        void set(void)
        {
            j = 0;
            baseClt.clear();
        }
        ex expr_visitor(const ex& e);
        ~funcSubs(){}
    };

    /** Applying the simplification rules x^(3/2)=x*x^(1/2)  **/
    class someMoreSimpRules:public map_function
    {
        int iNum;

    public:
        someMoreSimpRules(){}
        ex expr_visitor(const ex& e);
        ~someMoreSimpRules(){}
    };


    extern simplifyc Simplify;
    extern Collect_common_factorsc Collect_common_factors; // This collect common numerical factors inside base of fractional power.
    extern numSimplify numSimplifye;
    extern arguSimplify arguSimplifye;
    extern expandinv expandinve;
    extern fracPowBasSubs fracPowBasSubsE;
    extern funcSubs funcSubsE;
    extern someMoreSimpRules someMoreSimpRulesE;

    ex Collect_common_factor(const ex& e);
    ///**this Simplify2 function simplify only the algebraic expressions containing
    //fractional power of F,F_,Fd_,X_,Y_; such as: simplify (F^(1/3))^4 as F^(4/3).**/
    //ex Simplify2(const ex& expr_);

    ex simplify(const ex& expr_, unsigned int rules = simplify_options::FuncSimp);
    ex fullsimplify(const ex& expr_, unsigned int rules = simplify_options::FuncSimp);
    inline ex fullSimplify(const ex& expr_, unsigned int rules = simplify_options::FuncSimp)
    {
        ex prev, curr = expr_;

        do
        {
            std::cout<<curr<<std::endl;
             std::cout<<prev<<std::endl;
            prev = expand(normal(curr));
            curr = Simplify(prev, rules);
        }while(prev!=curr);

        return  curr;
    }
}

#endif // SIMPLIFY_H_INCLUDED
