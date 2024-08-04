#ifndef IMAP_FUNCTIONS_H
#define IMAP_FUNCTIONS_H

// Created by Cython when providing 'public api' keywords
#include <Python.h>
#include "ginacsym/ginacsympy_api.h"
#include "ginacsym/basic.h"

namespace ginacsym {

class imap_function : public map_function {
public:
    PyObject *m_obj;

    imap_function(PyObject *obj);
    virtual ~imap_function();
    virtual ex expr_visitor(const ex & e);
};

} /* namespace elps */

#endif // IMAP_FUNCTIONS_H
