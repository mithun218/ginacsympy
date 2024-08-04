/** @file version.h
 *
 *  ginacsym library version information. */

/*
 *  ginacsym Copyright (C) 1999-2023 Johannes Gutenberg University Mainz, Germany
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

#ifndef GINACSYM_VERSION_H
#define GINACSYM_VERSION_H

/* Major version of ginacsym */
#define GINACSYMLIB_MAJOR_VERSION 1

/* Minor version of ginacsym */
#define GINACSYMLIB_MINOR_VERSION 0

/* Micro version of ginacsym */
#define GINACSYMLIB_MICRO_VERSION 0

// ginacsym library version information. It has very little to do with ginacsym
// version number. In particular, library version is OS dependent. 
//
// When making releases, do
// 1. Increment GINACSYM_LT_REVISION
// 2. If any interfaces have been added, removed, or changed since the last
//    release, increment GINACSYM_LT_CURRENT and set GINACSYM_LT_REVISION to 0.
// 3. If any interfaces have been added since the last release, increment
//    GINACSYM_LT_AGE.
// 4. If any interfaces have been removed since the last release, set 
//    GINACSYM_LT_AGE to 0.
//
// Please note: the libtool naming scheme cannot guarantee that on all
// systems, the numbering is consecutive. It only guarantees that it is
// increasing. This doesn't matter, though: there is not incurred cost
// for numbers that are omitted, except for shrinking the available space
// of leftover numbers. Not something we need to worry about yet. ;-)
//
// On Linux, the SONAME is libginac.so.$(GINACSYM_LT_CURRENT)-$(GINACSYM_LT_AGE).
//
// TODO, when breaking the SONAME:
//  * change matrix inverse to use default argument (twice)
//  * check for interfaces marked as deprecated
#define GINACSYM_LT_CURRENT  1
#define GINACSYM_LT_REVISION 1
#define GINACSYM_LT_AGE      1

/*
 * ginacsym archive file version information.
 *
 * The current archive version is GINACSYMLIB_ARCHIVE_VERSION. This is
 * the version of archives created by the current version of ginacsym.
 * Archives version (GINACSYMLIB_ARCHIVE_VERSION - GINACSYMLIB_ARCHIVE_AGE)
 * thru * GINACSYMLIB_ARCHIVE_VERSION can be read by current version
 * of ginacsym.
 *
 * Backward compatibility notes:
 * If new properties have been added:
 *	GINACSYMLIB_ARCHIVE_VERSION += 1
 *	GINACSYMLIB_ARCHIVE_AGE += 1
 * If backwards compatibility has been broken, i.e. some properties
 * has been removed, or their type and/or meaning changed:
 *	GINACSYMLIB_ARCHIVE_VERSION += 1
 *	GINACSYMLIB_ARCHIVE_AGE = 0
 */
#define GINACSYMLIB_ARCHIVE_VERSION 3
#define GINACSYMLIB_ARCHIVE_AGE 3

#define GINACSYMLIB_STR_HELPER(x) #x
#define GINACSYMLIB_STR(x) GINACSYMLIB_STR_HELPER(x)
#define GINACSYMLIB_VERSION \
	GINACSYMLIB_STR(GINACSYMLIB_MAJOR_VERSION) "." \
	GINACSYMLIB_STR(GINACSYMLIB_MINOR_VERSION) "." \
	GINACSYMLIB_STR(GINACSYMLIB_MICRO_VERSION)

namespace ginacsym {

extern const int version_major;
extern const int version_minor;
extern const int version_micro;

} // namespace ginacsym

#endif // ndef GINACSYM_VERSION_H
