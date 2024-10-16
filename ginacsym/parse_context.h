/** @file parse_context.h
 *
 *  Interface to parser context. */

/*
 *  GiNaC Copyright (C) 1999-2024 Johannes Gutenberg University Mainz, Germany
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

#ifndef GINACSYM_PARSE_CONTEXT_H
#define GINACSYM_PARSE_CONTEXT_H

#include "ex.h"
#include "symbol.h"

#include <cstddef> // for size_t
#include <map>
#include <string>
#include <utility>

namespace ginacsym {

/**
 * Establishes correspondence between the strings and expressions.
 * The parser will create missing symbols (if not instructed otherwise,
 * in which case it fails if the expression contains unknown symbols).
 */
typedef std::map<std::string, ex> symtab;

/**
 * Find the symbol (or abbreviation) with the @a name in the symbol table @a syms.
 *
 * If symbol is missing and @a strict = false, insert it, otherwise
 * throw an exception.
 */
extern ex 
find_or_insert_symbol(const std::string& name, symtab& syms,
	              const bool strict);

/**
 * Function (or class ctor) prototype
 * .first is  the name of function(or ctor),
 * .second is the number of arguments (each of type ex)
 */
typedef std::pair<std::string, std::size_t> prototype;

/**
 * A (C++) function for reading functions and classes from the stream.
 *
 * The parser uses (an associative array of) such functions to construct
 * (ginacsym) classes and functions from a sequence of characters.
 */
class reader_func {
	enum { FUNCTION_PTR, GINACSYM_FUNCTION };
public:
	reader_func(ex (*func_)(const exvector& args))
		: type(FUNCTION_PTR), serial(0), func(func_) {}
	reader_func(unsigned serial_)
		: type(GINACSYM_FUNCTION), serial(serial_), func(nullptr) {}
	ex operator()(const exvector& args) const;
private:
	unsigned type;
	unsigned serial;
	ex (*func)(const exvector& args);
};



/**
 * Prototype table.
 *
 * If parser sees an expression which looks like a function call (e.g.
 * foo(x+y, z^2, t)), it looks up such a table to find out which
 * function (or class) corresponds to the given name and has the given
 * number of the arguments.
 *
 * N.B.
 *
 * 1. The function don't have to return a (ginacsym) function or class, it
 *    can return any expression.
 * 2. Overloaded functions/ctors are paritally supported, i.e. there might
 *    be several functions with the same name, but they should take different
 *    number of arguments.
 * 3. User can extend the parser via custom prototype tables. It's possible
 *    to read user defined classes, create abbreviations, etc.
 *
 * NOTE: due to a hack that allows user defined functions to be parsed, the map
 *       value of type reader_func is internally treated as an unsigned and not as a
 *       function pointer!! The unsigned has to correspond to the serial number of
 *       the defined ginacsym function.
 */
class PrototypeLess
{
public:
	bool operator()(const prototype& p1, const prototype& p2) const
	{
		int s = p1.first.compare(p2.first);
		if (s == 0) {
			if ((p1.second == 0) || (p2.second == 0)) return false;
			return p1.second < p2.second;
		}
		return s < 0;
	}
};
typedef std::map<prototype, reader_func, PrototypeLess> prototype_table;

/**
 * Default prototype table.
 *
 * It supports all defined ginacsym functions and "pow", "sqrt", and "power".
 */
extern const prototype_table& get_default_reader();
/**
 * Builtin prototype table.
 *
 * It supports only the builtin ginacsym functions and "pow", "sqrt", and "power".
 */
extern const prototype_table& get_builtin_reader();

} // namespace ginacsym

#endif // GINACSYM_PARSE_CONTEXT_H
