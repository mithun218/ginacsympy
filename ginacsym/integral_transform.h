#ifndef INTEGRAL_TRANSFORM_H
#define INTEGRAL_TRANSFORM_H

#include "basic.h"
#include "ex.h"
#include "utility.h"

namespace ginacsym {

    //Calculate Laplace Transform
    ex laplace_transform(const ex &f, const ex &t, const ex &s, stepContext *ctx = nullptr);

    //Calculate Inverse Laplace Transform
    ex inverse_laplace_transform(const ex &F, const ex &s, const ex &t, stepContext *ctx = nullptr);

}

#endif // INTEGRAL_TRANSFORM_H
