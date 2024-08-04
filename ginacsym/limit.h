#ifndef LIMIT_H
#define LIMIT_H

#include "ex.h"
namespace ginacsym {

/***  """Computes the limit of ``e(z)`` at the point ``z0``.

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

ex limit(const ex& e, const ex& z, const ex& z0, const std::string& dir="+-");

}

#endif // LIMIT_H
