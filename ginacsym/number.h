#ifndef NUMBER_H
#define NUMBER_H

#include "ex.h"
#include <flint/fmpz.h>
#include <flint/fmpz_poly.h>
#include <string>


namespace ginacsym {

class flint_variablec
{
public:
    fmpz_t fmpz_1,fmpz_2;
    fmpz_mpoly_t fmpz_mpoly_1,fmpz_mpoly_2;
    flint_variablec() {fmpz_init(fmpz_1);fmpz_init(fmpz_2);}
    ~flint_variablec(){fmpz_clear(fmpz_1);fmpz_clear(fmpz_2);flint_cleanup_master();}
};

extern flint_variablec flint_variable;
//the digits of n in base b
std::string base_form(const ex& n, const int &b);

//test whether m is divisible by n
bool divisible(const ex& m, const ex& n);

////test whether m and n are coprime
//bool coprime_test(const ex& m, const ex& n);

//next smallest prime above n
ex next_prime(const ex& n);

//random prime generator less than and equal to n
ex random_prime(const ex& n);

}



#endif // NUMBER_H
