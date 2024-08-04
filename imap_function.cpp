#include "imap_functions.h"
#include "ginacsym/ex.h"
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <iostream>
#include "ginacsym/ginacsympy.h"

namespace ginacsym {

imap_function::imap_function(PyObject *obj): m_obj(obj) {
    // Provided by "elps_api.h"
    if (import_ginacsympy()) {
    } else {
        Py_XINCREF(this->m_obj);
    }
}

imap_function::~imap_function() {
    Py_XDECREF(this->m_obj);
}

ex imap_function::expr_visitor(const ex & e)
{
    if (this->m_obj) {
        int error;
        // Call a virtual overload, if it exists
        const ex tem=e;
        ex result = cy_call_func(this->m_obj,tem, (char*)"expr_visitor", &error);
        if (error)
            // Call parent method
            result = map_function::expr_visitor(e);
        return result;
    }
    // Throw error ?
//    return ex(0);
    throw std::invalid_argument("unable to overload virtual function.");
}

} /* namespace elps */
