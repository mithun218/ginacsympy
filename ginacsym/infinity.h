/** @file infinity.h
 *
 *  The value "Infinity".  */

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

#ifndef __GINAC_INFINITY_H__
#define __GINAC_INFINITY_H__

#include "basic.h"
#include "ex.h"
#include "relational.h"
#include "utils.h"


namespace ginacsym {

	
/** This class holds "infinity"
 *  It includes a direction (like -infinity).
 **/
class infinity : public basic
{
    GINAC_DECLARE_REGISTERED_CLASS(infinity, basic)

	// functions overriding virtual functions from base classes
public:
    infinity(const ex& new_sign):sign(new_sign){}
	bool info(unsigned inf) const override;
    //ex evalf(int level = 0, PyObject* parent=nullptr) const override;
    ex evalf() const override {return *this;}
	ex conjugate() const override;
	ex real_part() const override;
	ex imag_part() const override;

    infinity operator *=(const ex & rhs);
    infinity operator +=(const ex & rhs) const;

    ex get_sign() const{return sign;}
    bool compare_other_type(const ex & other,
                            relational::operators op) const;

protected:
	ex derivative(const symbol & s) const override;
	
	// non-virtual functions in this class
    void do_print(const print_context & c, unsigned level) const;
	void do_print_latex(const print_latex & c, unsigned level) const;

    void set_sign(const ex & new_sign);

private:
    ex sign;
	
};

extern const infinity Infinity;

} // namespace ginacsym

#endif // ndef __GINAC_INFINITY_H__
