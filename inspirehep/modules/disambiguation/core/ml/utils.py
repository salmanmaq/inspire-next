# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2017 CERN.
#
# INSPIRE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# INSPIRE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with INSPIRE. If not, see <http://www.gnu.org/licenses/>.
#
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction

"""Disambiguation core ML utils."""

from __future__ import absolute_import, division, print_function

from beard.utils import (
    given_name,
    given_name_initial,
    name_initials,
    normalize_name,
)


def get_author_full_name(s):
    """Get author full name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized author name
    """
    v = s["author_name"]
    v = normalize_name(v) if v else ""
    return v


def get_first_given_name(s):
    """Get author first given name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Author's first given name
    """
    v = given_name(s["author_name"], 0)
    return v


def get_second_given_name(s):
    """Get author second given name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Author's second given name
    """
    v = given_name(s["author_name"], 1)
    return v


def get_surname(s):
    return s['author_name'].split(" ")[0].split(",")[0]


def get_first_initial(s):
    v = given_name_initial(s["author_name"], 0)
    try:
        return v
    except IndexError:
        return ""


def get_second_initial(s):
    """Get author second given name's initial from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Second given name's initial. Empty string in case it's not available.
    """
    v = given_name_initial(s["author_name"], 1)
    try:
        return v
    except IndexError:
        return ""


def get_author_other_names(s):
    """Get author other names from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized other author names
    """
    v = s["author_name"]
    v = v.split(",", 1)
    v = normalize_name(v[1]) if len(v) == 2 else ""
    return v


def get_author_initials(s):
    """Get author initials from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Initials, not separated
    """
    v = s["author_name"]
    v = v if v else ""
    v = "".join(name_initials(v))
    return v


def get_author_affiliation(s):
    """Get author affiliation from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized affiliation name
    """
    v = s["author_affiliation"]
    v = normalize_name(v) if v else ""
    return v


def get_title(s):
    """Get publication's title from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Title of the publication
    """
    v = s["publication"]["title"]
    v = v if v else ""
    return v


def get_abstract(s):
    """Get author full name from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Normalized author name
    """
    v = s["publication"]["abstract"]
    v = v if v else ""
    return v


def get_coauthors(s):
    """Get coauthors from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Coauthors ids separated by a space
    """
    v = s["publication"]["authors"]
    v = " ".join(v)
    return v


def get_coauthors_from_range(s, range=10):
    """Get coauthors from the signature.

    Only the signatures from the range-neighbourhood of the given signature
    will be selected. Signatures on the paper are ordered (although they don't
    have to be sorted!), and the distance between signatures is defined
    as absolute difference of the indices.

    The function was introduced due to the high memory usage of
    a simple version.

    Parameters
    ----------
    :param s: dict
        Signature
    :param range: integer
        The maximum distance for the signatures between the author and his
        coauthor.

    Returns
    -------
    :returns: string
        Coauthors ids separated by a space
    """
    v = s["publication"]["authors"]
    try:
        index = v.index(s["author_name"])
        v = " ".join(v[max(0, index - range):min(len(v), index + range)])
        return v
    except ValueError:
        v = " ".join(v)
        return v


def get_keywords(s):
    """Get keywords from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Keywords separated by a space
    """
    v = s["publication"]["keywords"]
    v = " ".join(v)
    return v


def get_topics(s):
    """Get topics from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Topics separated by a space
    """
    v = s["publication"]["topics"]
    v = " ".join(v)
    return v


def get_collaborations(s):
    """Get collaborations from the signature.

    Parameters
    ----------
    :param s: dict
        Signature

    Returns
    -------
    :returns: string
        Collaboations separated by a space
    """
    v = s["publication"]["collaborations"]
    v = " ".join(v)
    return v


def group_by_signature(r):
    """Grouping function for ``PairTransformer``.

    Parameters
    ----------
    :param r: iterable
        signature in a singleton.

    Returns
    -------
    :returns: string
        Signature id
    """
    return r[0]["signature_uuid"]
