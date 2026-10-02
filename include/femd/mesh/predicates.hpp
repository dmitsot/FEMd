//
//  predicates.hpp  --  sign of orient2d and incircle, filtered then exact.
//
//  Both determinants are first evaluated in double together with the standard
//  forward error bound for the expression.  When |det| clears the bound the
//  sign is certain and is returned at once, which is the path taken for
//  essentially every call.  Otherwise the determinant is recomputed in
//  double-double arithmetic (Dekker/Knuth error-free transformations, ~32
//  significant digits), which for mesh coordinates is exact.
//
//  This matters more than it looks.  Bowyer-Watson digs a cavity by asking
//  incircle() about a set of triangles; if two calls disagree about the same
//  circle because of rounding, the cavity is not star-shaped and the
//  retriangulation produces overlapping or inverted triangles.  The original C
//  meshgen used plain double comparisons against a 1e-12 tolerance, which is
//  exactly the failure mode this header removes.
//
#ifndef FEMD_MESH_PREDICATES_HPP
#define FEMD_MESH_PREDICATES_HPP

#include <cmath>

namespace femd {
namespace pred {

// ---- double-double: an unevaluated sum hi + lo with |lo| <= ulp(hi)/2 ------

struct DD {
    double hi = 0.0, lo = 0.0;
    DD() = default;
    DD(double h, double l) : hi(h), lo(l) {}
    explicit DD(double h) : hi(h), lo(0.0) {}
};

/// Exact sum of two doubles (Knuth's two-sum): a + b == s + e, always.
inline DD two_sum(double a, double b)
{
    double s = a + b;
    double bb = s - a;
    return DD(s, (a - (s - bb)) + (b - bb));
}

/// Exact difference of two doubles.
inline DD two_diff(double a, double b)
{
    double s = a - b;
    double bb = s - a;
    return DD(s, (a - (s - bb)) - (b + bb));
}

/// Dekker split of a double into two 26-bit halves.
inline void split(double a, double &hi, double &lo)
{
    double t = 134217729.0 * a;   // 2^27 + 1
    hi = t - (t - a);
    lo = a - hi;
}

/// Exact product of two doubles: a*b == p + e, always.
inline DD two_prod(double a, double b)
{
    double p = a * b;
    double ah, al, bh, bl;
    split(a, ah, al);
    split(b, bh, bl);
    double e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    return DD(p, e);
}

inline DD add(const DD &a, const DD &b)
{
    DD s = two_sum(a.hi, b.hi);
    double lo = s.lo + (a.lo + b.lo);
    return two_sum(s.hi, lo);
}

inline DD sub(const DD &a, const DD &b) { return add(a, DD(-b.hi, -b.lo)); }

inline DD mul(const DD &a, const DD &b)
{
    DD p = two_prod(a.hi, b.hi);
    double lo = p.lo + (a.hi * b.lo + a.lo * b.hi);
    return two_sum(p.hi, lo);
}

inline int sign(const DD &a) { return a.hi > 0.0 ? 1 : (a.hi < 0.0 ? -1 : (a.lo > 0.0 ? 1 : (a.lo < 0.0 ? -1 : 0))); }

// ---- error bounds (Shewchuk's constants, epsilon = 2^-53) ------------------

inline constexpr double kEps        = 1.1102230246251565e-16;  // 2^-53
inline constexpr double kOrientBound  = (3.0 + 16.0 * kEps) * kEps;
inline constexpr double kIncircleBound = (10.0 + 96.0 * kEps) * kEps;

// ---- the predicates --------------------------------------------------------

/**
 * @brief Sign of the area of the triangle (a, b, c), times two.
 * @return +1 if a, b, c turn counter-clockwise, -1 if clockwise, 0 if collinear.
 *
 * The sign is exact: a 0 means the three points really are collinear, not
 * merely close to it.
 */
inline int orient2d(double ax, double ay, double bx, double by, double cx, double cy)
{
    double acx = ax - cx, acy = ay - cy;
    double bcx = bx - cx, bcy = by - cy;
    double l = acx * bcy, r = acy * bcx;
    double det = l - r;

    double s = std::abs(l) + std::abs(r);
    if (det > kOrientBound * s) return 1;
    if (-det > kOrientBound * s) return -1;
    if (s == 0.0) return 0;

    // exact path
    DD dacx = two_diff(ax, cx), dacy = two_diff(ay, cy);
    DD dbcx = two_diff(bx, cx), dbcy = two_diff(by, cy);
    return sign(sub(mul(dacx, dbcy), mul(dacy, dbcx)));
}

/**
 * @brief Where d lies relative to the circle through a, b, c.
 * @return +1 if d is strictly inside, -1 if strictly outside, 0 if cocircular.
 * @note Assumes a, b, c are counter-clockwise; reverse the sign if they are not.
 */
inline int incircle(double ax, double ay, double bx, double by,
                    double cx, double cy, double dx, double dy)
{
    double adx = ax - dx, ady = ay - dy;
    double bdx = bx - dx, bdy = by - dy;
    double cdx = cx - dx, cdy = cy - dy;

    double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
    double cdxady = cdx * ady, adxcdy = adx * cdy;
    double adxbdy = adx * bdy, bdxady = bdx * ady;

    double alift = adx * adx + ady * ady;
    double blift = bdx * bdx + bdy * bdy;
    double clift = cdx * cdx + cdy * cdy;

    double det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);

    double perm = (std::abs(bdxcdy) + std::abs(cdxbdy)) * alift
                + (std::abs(cdxady) + std::abs(adxcdy)) * blift
                + (std::abs(adxbdy) + std::abs(bdxady)) * clift;

    if (det > kIncircleBound * perm) return 1;
    if (-det > kIncircleBound * perm) return -1;
    if (perm == 0.0) return 0;

    // exact path
    DD Ax = two_diff(ax, dx), Ay = two_diff(ay, dy);
    DD Bx = two_diff(bx, dx), By = two_diff(by, dy);
    DD Cx = two_diff(cx, dx), Cy = two_diff(cy, dy);

    DD AL = add(mul(Ax, Ax), mul(Ay, Ay));
    DD BL = add(mul(Bx, Bx), mul(By, By));
    DD CL = add(mul(Cx, Cx), mul(Cy, Cy));

    DD t1 = sub(mul(Bx, Cy), mul(Cx, By));
    DD t2 = sub(mul(Cx, Ay), mul(Ax, Cy));
    DD t3 = sub(mul(Ax, By), mul(Bx, Ay));

    return sign(add(add(mul(AL, t1), mul(BL, t2)), mul(CL, t3)));
}

} // namespace pred
} // namespace femd

#endif // FEMD_MESH_PREDICATES_HPP
