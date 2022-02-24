#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <iosfwd>
#include <type_traits>

#if __x86_64
#ifdef USE_SSE
#include "immintrin.h"
#endif
#endif

using f32 = float;
using f64 = double;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using i8  = int8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;

template <int>
struct vec;

template <int... n>
struct swizz;

template <int r, int c>
struct mat;

template <class, int>
struct int_vec;

using vec2 = vec<2>;
using vec3 = vec<3>;
using vec4 = vec<4>;

using mat2 = mat<2, 2>;
using mat3 = mat<3, 3>;
using mat4 = mat<4, 4>;

using ivec2 = int_vec<i32, 2>;
using ivec3 = int_vec<i32, 3>;
using ivec4 = int_vec<i32, 4>;
using uvec2 = int_vec<u32, 2>;
using uvec3 = int_vec<u32, 3>;
using uvec4 = int_vec<u32, 4>;

template <int...>
static constexpr bool is_unique = true;

template <int a, int... t>
static constexpr bool is_unique<a, t...> = ((a != t) && ...) && is_unique<t...>;

template <int...>
struct Seqx;

template <int h, int... t>
struct Seq0 : Seq0<h - 1, h, t...>
{
};

template <int... t>
struct Seq0<-1, t...>
{
    using Seq = Seqx<t...>;
};

template <int n>
using Seq = typename Seq0<n - 1>::Seq;

template <class, class>
struct Concat;

template <int... a, int... b>
struct Concat<Seqx<a...>, Seqx<b...>>
{
    using Type = swizz<a..., (b + sizeof...(a) + 1)...>;
};

template <int s, int n>
requires(s > n) using Removed = typename Concat<Seq<n>, Seq<s - n - 1>>::Type;

template <int... t>
struct Seqx
{
    static constexpr inline int n = sizeof...(t);

    using vt = vec<n>;
    using mt = mat<n, n>;

    static constexpr inline void assign(auto& l, vt r)
    {
        ((l[t] = r[t]), ...);
    }
    static constexpr inline vt cast(auto v)
    {
        return vt(v[t]...);
    }

#if __x86_64 && USE_SSE
    static constexpr inline vt add(vt l, vt r) requires(n == 4)
    {
        return _mm_add_ps(l.xmm, r.xmm);
    }
    static constexpr inline vt sub(vt l, vt r) requires(n == 4)
    {
        return _mm_sub_ps(l.xmm, r.xmm);
    }
    static constexpr inline vt mul(vt l, vt r) requires(n == 4)
    {
        return _mm_mul_ps(l.xmm, r.xmm);
    }
    static constexpr inline vt div(vt l, vt r) requires(n == 4)
    {
        return _mm_div_ps(l.xmm, r.xmm);
    }
    static constexpr inline vt add(vt l, vt r) requires(n != 4)
    {
        return vt(l[t] + r[t]...);
    }
    static constexpr inline vt sub(vt l, vt r) requires(n != 4)
    {
        return vt(l[t] - r[t]...);
    }
    static constexpr inline vt mul(vt l, vt r) requires(n != 4)
    {
        return vt(l[t] * r[t]...);
    }
    static constexpr inline vt div(vt l, vt r) requires(n != 4)
    {
        return vt(l[t] / r[t]...);
    }
#else
    static constexpr inline vt add(vt l, vt r)
    {
        return vt(l[t] + r[t]...);
    }
    static constexpr inline vt sub(vt l, vt r)
    {
        return vt(l[t] - r[t]...);
    }
    static constexpr inline vt mul(vt l, vt r)
    {
        return vt(l[t] * r[t]...);
    }
    static constexpr inline vt div(vt l, vt r)
    {
        return vt(l[t] / r[t]...);
    }
#endif

#undef REQUIRES_NONPACKED

    static constexpr inline vt neg(vt v)
    {
        return vt{-v[t]...};
    }

    template <bool odd, int s, int i, int j>
    static constexpr inline f32 idx(auto m)
    {
        if constexpr (i < s)
        {
            constexpr int C = n - j;
            constexpr int R = n - (i + j);
            static_assert(R < sizeof(m[0]) / 4);
            static_assert(C < sizeof(m) / sizeof(m[0]));
            const f32 x = m[C][R] - m[R][C];
            return odd ? -x : x;
        }
        else
            return idx<odd, s - 1, i + 1 - s, j + 1>(m);
    }

    template <int x>
    requires((x * x - x) / 2 == n) static constexpr inline vt wedge(mat<x, x> m)
    {
        return vt(Seq<x>::template idx<t & 1, x, t + 1, 1>(m)...);
    }

    template <class Swizz>
    static constexpr inline f32 sub_mat(mat<n + 1, n + 1> m)
    {
        return mt((Swizz&)m[t + 1]...).det();
    }

    static constexpr inline f32 det(mt m)
    {
        return alternating_dot(
            m[0], vt(Seq<n - 1>::template sub_mat<Removed<n, t>>(m)...));
    }

    static constexpr inline mt outer(vt l, vt r)
    {
        return mt{l * r[t]...};
    }

    static constexpr inline vt clamp(vt v, vt l, vt r)
    {
        return vt(v[t] < l[t] ? l[t] : v[t] > r[t] ? r[t]
                                                   : v[t]...);
    }

    static constexpr inline f32 dot(vt l, vt r)
    {
        return ((l[t] * r[t]) + ...);
    }

    static constexpr inline f32 alternating_dot(vt l, vt r)
    {
        return ((t & 1 ? -(l[t] * r[t]) : l[t] * r[t]) + ...);
    }

    static std::ostream& stream(std::ostream& s, vt v)
    {
        ((v[t] = (fabs(v[t]) < 0.00001f ? 0.f : v[t])), ...);
        return ((s << (v[t] >= 0 ? " " : "") << v[t] << ' '), ...) << "\b\n";
    }

    template <int a>
    static std::ostream& stream(std::ostream& s, mat<a, n> m)
    {
        return ((s << m[t]), ...) << "\b\n";
    }

    template <int r>
    static constexpr inline vec<r> mul(mat<r, n> m, vt v)
    {
        return ((m[t] * v[t]) + ...);
    }

    template <int c>
    static constexpr inline vt mul(vec<c> v, mat<c, n> m)
    {
        return vt{dot(m[t], v)...};
    }

    template <int a, int b>
    static constexpr inline mat<a, n> mul(mat<a, b> l, mat<b, n> r)
    {
        return mat<a, n>{(l * r[t])...};
    }

    template <int a>
    static constexpr inline mat<a, n> mul(mat<a, n> l, f32 r)
    {
        return mat<a, n>{l[t] * r...};
    }

    template <int a>
    static constexpr inline mat<a, n> add(mat<a, n> l, mat<a, n> r)
    {
        return mat<a, n>{l[t] + r[t]...};
    }

    template <int a>
    static constexpr inline mat<a, n> sub(mat<a, n> l, mat<a, n> r)
    {
        return mat<a, n>{(l[t] - r[t])...};
    }

    template <int a, int i>
    static constexpr inline void write(mat<a, n>& m, vt v)
    {
        ((m[t][i] = v[t]), ...);
    }

    template <int a>
    requires(a > n) static constexpr inline vt down_cast(vec<a> v)
    {
        return vt(v[t]...);
    }

    template <int a>
    requires(a > n) static constexpr inline mt down_cast(mat<a, a> m)
    {
        return mt(down_cast(m[t])...);
    }

    template <int a>
    requires(a < n) static constexpr inline vt up_cast(vec<a> v)
    {
        return vt((t < a ? v[t] : 0.f)...);
    }

    template <int a>
    requires(a < n) static constexpr inline mt up_cast(mat<a, a> m)
    {
        return mt((t < a ? up_cast(m[t]) : mt(1.f)[t])...);
    }

    template <int a>
    static constexpr inline mat<n, a> tpos(mat<a, n> m)
    {
        mat<n, a> re;
        ((Seq<a>::template write<n, t>(re, m[t])), ...);
        return re;
    }

    static constexpr inline void fill(mt& m, f32 x)
    {
        ((m[t][t] = x), ...);
    }
    static constexpr inline void fill(vt& v, f32 x)
    {
        ((v[t] = x), ...);
    }
};

template <int h, int... n>
static constexpr inline int swizz_max =
    h > swizz_max<n...> ? h : swizz_max<n...>;

template <int h, int... n>
static constexpr inline int swizz_min =
    h < swizz_min<n...> ? h : swizz_min<n...>;

template <int n>
static constexpr inline int swizz_max<n> = n;

template <int n>
static constexpr inline int swizz_min<n> = n;

template <bool s>
static constexpr inline f32 set_float_sign(f32 x)
{
    (i32&)x ^= (s << 31);
    return x;
}

template <int x, int y, int z, int w>
static constexpr inline int mask()
{
    return _MM_SHUFFLE(w, z, y, x);
}

template <int... n>
struct swizz
{
    static constexpr inline int  ns    = sizeof...(n);
    static constexpr inline int  idx[] = {n...};
    static constexpr inline int  mx    = swizz_max<n...>;
    static constexpr inline int  mn    = swizz_min<n...>;
    static constexpr inline bool gapless =
        ((mx - mn + 1) * (mx + mn)) / 2 == (n + ...);

    f32 dat[mx + 1];
    using vt = vec<sizeof...(n)>;

#if __x86_64 && USE_SSE
    constexpr inline operator vt() const requires(ns == 4)
    {
        constexpr const int xmask = mask<n...>();
        __m128              xmm   = _mm_loadu_ps(dat);
        return _mm_shuffle_ps(xmm, xmm, xmask);
    }

    constexpr inline operator vt() const requires(!(ns == 4))
    {
        return vt{dat[n]...};
    }
#else
    constexpr inline operator vt() const
    {
        return vt{dat[n]...};
    }
#endif

    template <int... t>
    constexpr inline swizz&
    operator=(swizz<t...> v) requires(ns == sizeof...(t) && is_unique<n...>)
    {
        ((dat[n] = v.dat[t]), ...);
        return *this;
    }

    constexpr inline swizz& operator=(swizz v) requires(is_unique<n...>)
    {
        ((dat[n] = v.dat[n]), ...);
        return *this;
    }

    constexpr inline swizz& operator=(vt v) requires(is_unique<n...>)
    {
        Seq<ns>::assign(*this, v);
        return *this;
    }

    template <bool... s>
    requires(sizeof...(s) == ns) constexpr inline vt sign() const
    {
        return vt((set_float_sign<s>(dat[n]))...);
    }

    constexpr inline f32& operator[](int i)
    {
        return dat[idx[i]];
    };
    constexpr inline f32 operator[](int i) const
    {
        return dat[idx[i]];
    };

    constexpr inline vt operator*(vt r) const
    {
        return vt(*this) * r;
    }
    constexpr inline vt operator/(vt r) const
    {
        return vt(*this) / r;
    }
    constexpr inline vt operator+(vt r) const
    {
        return vt(*this) + r;
    }
    constexpr inline vt operator-(vt r) const
    {
        return vt(*this) - r;
    }
    constexpr inline vt operator*(f32 r) const
    {
        return vt(*this) * r;
    }
    constexpr inline vt operator/(f32 r) const
    {
        return vt(*this) / r;
    }
    constexpr inline vt operator-() const
    {
        return -vt(*this);
    }
    constexpr inline friend vt operator*(f32 l, swizz r)
    {
        return l * vt(r);
    }
    constexpr inline friend vt operator/(f32 l, swizz r)
    {
        return l / vt(r);
    }
    constexpr inline swizz& operator*=(vt r)
    {
        return *this = vt(*this) * r;
    }
    constexpr inline swizz& operator/=(vt r)
    {
        return *this = vt(*this) / r;
    }
    constexpr inline swizz& operator+=(vt r)
    {
        return *this = vt(*this) + r;
    }
    constexpr inline swizz& operator-=(vt r)
    {
        return *this = vt(*this) - r;
    }
    constexpr inline swizz& operator*=(f32 r)
    {
        return *this = vt(*this) * r;
    }
    constexpr inline swizz& operator/=(f32 r)
    {
        return *this = vt(*this) / r;
    }
    constexpr inline f32 dot(vt r) const
    {
        return vt(*this).dot(r);
    }
    constexpr inline f32 len() const
    {
        return vt(*this).len();
    }
    constexpr inline f32 len2() const
    {
        return vt(*this).len2();
    }
    constexpr inline mat<ns, ns> outer(vt r) const
    {
        return vt(*this).outer(r);
    }
    constexpr inline vt norm() const
    {
        return vt(*this).norm();
    }
    constexpr inline swizz& normalize()
    {
        return *this /= len();
    }
    constexpr inline vec<(ns * ns - ns) / 2> wedge(vt r) const
    {
        return vt(*this).wedge(r);
    }
    constexpr inline vt clamp(vt min, vt max) const
    {
        return vt(*this).clamp(min, max);
    }

    friend std::ostream& operator<<(std::ostream& s, swizz v)
    {
        return s << vt(v);
    }
};

template <int n>
struct vec_data
{
    f32 dat[n];

    template <int... t>
    requires(swizz_max<t...> <
             n) constexpr inline vec<sizeof...(t)> shuff() const
    {
        return *reinterpret_cast<const swizz<t...>*>(this);
    }
};

#include "vec_data.inl"

template <int r, int c>
struct mat_data
{
    vec<r> dat[c];
};

template <int r>
struct mat_data<r, 2>
{
    union {
        vec<r> dat[2];
        struct
        {
            vec<r> x, y;
        };
    };
};

template <int r>
struct mat_data<r, 3>
{
    union {
        vec<r> dat[3];
        struct
        {
            vec<r> x, y, z;
        };
    };
};

template <int r>
struct mat_data<r, 4>
{
    union {
        vec<r> dat[4];
        struct
        {
            vec<r> x, y, z, w;
        };
    };
};

template <class T, int n>
struct int_vec_data
{
    T dat[n];
};

template <class T>
struct int_vec_data<T, 2>
{
    union {
        T dat[2];
        T x, y;
    };
};

template <class T>
struct int_vec_data<T, 3>
{
    union {
        T dat[3];
        T x, y, z;
    };
};

template <class T>
struct int_vec_data<T, 4>
{
    union {
        T dat[4];
        T x, y, z, w;
    };
};

template <class T, int n>
constexpr inline bool operator==(int_vec_data<T, n> const& l,
                                 int_vec_data<T, n> const& r)
{
    return memcmp(l.dat, r.dat, sizeof(T) * n) == 0;
}

template <class T, int n>
struct int_vec : int_vec_data<T, n>
{
    using int_vec_data<T, n>::dat;

    constexpr inline bool operator==(int_vec const&) const = default;
    constexpr inline T    operator[](int i) const
    {
        return dat[i];
    }
    constexpr inline T& operator[](int i)
    {
        return dat[i];
    }
    constexpr inline operator vec<n>() const
    {
        return Seq<n>::cast(*this);
    }
};

template <class U, int n>
struct ctor_arg
{
    using V = std::remove_reference_t<U>;

    static constexpr inline int Val()
    {
        if constexpr (std::is_convertible_v<V, f32>)
            return 1;
        if constexpr (std::is_convertible_v<V, vec<n>>)
            return n;
        if constexpr (std::is_same_v<V, vec<n>>)
            return n;
        if constexpr (n == 1)
            return 0;
        else
            return ctor_arg<V, n - 1>::val;
    }

    static constexpr inline int val = Val();

    using type = std::conditional_t<val == 1, f32, vec<val>>;

    static constexpr inline type& bump(f32*& p)
    {
        type* t = (type*)p;
        p += val;
        return *t;
    }
};

template <int n>
struct vec : vec_data<n>
{
    using base = vec_data<n>;
    using base::dat;

    template <class T>
    static constexpr inline int ctor_val = ctor_arg<T, n>::Val();

    template <class... T>
    constexpr inline vec(T&&... arg) requires(sizeof...(T) >= 2 &&
                                              ((0 < ctor_val<T>)&&...) &&
                                              (ctor_val<T> + ...) == n)
    {
        f32* p = dat;
        ((ctor_arg<T, n>::bump(p) = arg), ...);
    }

    constexpr inline vec() = default;
    constexpr inline explicit vec(f32 x)
    {
        Seq<n>::fill(*this, x);
    }
    constexpr inline vec(vec const&) = default;
    constexpr inline vec& operator   =(vec const& v)
    {
        memcpy(this, &v, sizeof v);
        return *this;
    };

#if __x86_64 && USE_SSE
    constexpr inline vec(__m128 v) requires(n == 4)
    {
        this->xmm = v;
    }
#endif

    template <int r>
    requires(r > n) constexpr inline vec<r> upcast() const
    {
        return Seq<r>::template up_cast<n>(*this);
    }

    template <int r>
    requires(r < n) constexpr inline vec<r> downcast() const
    {
        return Seq<r>::template down_cast<n>(*this);
    }

    constexpr inline f32& operator[](int i)
    {
        return dat[i];
    };
    constexpr inline f32 operator[](int i) const
    {
        return dat[i];
    };
    constexpr inline f32* begin()
    {
        return dat;
    }
    constexpr inline f32* end()
    {
        return dat + n;
    }
    constexpr inline const f32* begin() const
    {
        return dat;
    }
    constexpr inline const f32* end() const
    {
        return dat + n;
    }
    constexpr inline vec operator-() const
    {
        return Seq<n>::neg(*this);
    }

    static constexpr inline vec (*add)(vec, vec) = Seq<n>::add;
    static constexpr inline vec (*sub)(vec, vec) = Seq<n>::sub;
    static constexpr inline vec (*mul)(vec, vec) = Seq<n>::mul;
    static constexpr inline vec (*div)(vec, vec) = Seq<n>::div;

    constexpr inline vec operator*(vec r) const
    {
        return mul(*this, r);
    }
    constexpr inline vec operator/(vec r) const
    {
        return div(*this, r);
    }
    constexpr inline vec operator+(vec r) const
    {
        return add(*this, r);
    }
    constexpr inline vec operator-(vec r) const
    {
        return sub(*this, r);
    }
    constexpr inline vec operator*(f32 r) const
    {
        return mul(*this, vec(r));
    }
    constexpr inline vec operator/(f32 r) const
    {
        return div(*this, vec(r));
    }
    constexpr inline friend vec operator*(f32 l, vec r)
    {
        return mul(vec(l), r);
    }
    constexpr inline friend vec operator/(f32 l, vec r)
    {
        return div(vec(l), r);
    }

    constexpr inline vec& operator*=(vec r)
    {
        return *this = *this * r;
    }
    constexpr inline vec& operator/=(vec r)
    {
        return *this = *this / r;
    }
    constexpr inline vec& operator+=(vec r)
    {
        return *this = *this + r;
    }
    constexpr inline vec& operator-=(vec r)
    {
        return *this = *this - r;
    }
    constexpr inline vec& operator*=(f32 r)
    {
        return *this = *this * r;
    }
    constexpr inline vec& operator/=(f32 r)
    {
        return *this = *this / r;
    }
    constexpr inline f32 dot(vec r) const
    {
        return Seq<n>::dot(*this, r);
    }

    constexpr inline f32 alternating_dot(vec r) const
    {
        return Seq<n>::alternating_dot(*this, r);
    }

    constexpr inline f32 len() const
    {
        return sqrtf(len2());
    }
    constexpr inline f32 len2() const
    {
        return dot(*this);
    }
    constexpr inline mat<n, n> outer(vec r) const
    {
        return Seq<n>::outer(*this, r);
    }
    constexpr inline vec norm() const
    {
        return *this / len();
    }
    constexpr inline vec& normalize()
    {
        return *this /= len();
    }
    constexpr inline vec<(n * n - n) / 2> wedge(vec r) const
    {
        return Seq<(n * n - n) / 2>::wedge(outer(r));
    }
    constexpr inline vec clamp(vec min, vec max) const
    {
        return Seq<n>::clamp(*this, min, max);
    }
    friend std::ostream& operator<<(std::ostream& s, vec v)
    {
        return Seq<n>::stream(s, v);
    }
};

template <int r, int c>
struct mat : mat_data<r, c>
{
    using base = mat_data<r, c>;
    using base::dat;

    constexpr inline vec<r>& operator[](int i)
    {
        return dat[i];
    };
    constexpr inline vec<r> const& operator[](int i) const
    {
        return dat[i];
    };

    template <int n>
    constexpr inline operator mat<n, n>() const requires(r == c && r > n)
    {
        return Seq<n>::down_cast(*this);
    }

    template <int n>
    constexpr inline operator mat<n, n>() const requires(r == c && r < n)
    {
        return Seq<n>::up_cast(*this);
    }

    constexpr inline mat()           = default;
    constexpr inline mat(mat const&) = default;
    constexpr inline mat& operator   =(mat const& m)
    {
        memcpy(dat, &m, sizeof m);
        return *this;
    }

    constexpr inline explicit mat(f32 x) requires(r == c)
        : base{}
    {
        Seq<r>::fill(*this, x);
    }

    template <class... T>
    constexpr inline mat(T&&... arg) requires((std::is_convertible_v<T, vec<r>> &&
                                               ...) &&
                                              sizeof...(T) == c)
    {
        vec<r>* p = dat;
        (((*p++ = arg)), ...);
    }

    constexpr inline vec<r> operator*(vec<r> v) const
    {
        return Seq<c>::mul(*this, v);
    }
    constexpr inline friend vec<r> operator*(vec<c> v, mat m)
    {
        return Seq<c>::mul(v, m);
    }

    template <int n>
    constexpr inline mat<r, n> operator*(mat<c, n> m) const
    {
        return Seq<n>::mul(*this, m);
    }

    constexpr inline mat operator*(f32 x) const
    {
        return Seq<c>::mul(*this, x);
    }
    constexpr inline friend mat operator*(f32 x, mat m)
    {
        return m * x;
    }
    constexpr inline mat operator+(mat m) const
    {
        return Seq<c>::add(*this, m);
    }
    constexpr inline mat operator-(mat m) const
    {
        return Seq<c>::sub(*this, m);
    }
    constexpr inline mat<c, r> tpos() const
    {
        return Seq<c>::tpos(*this);
    }

    constexpr inline f32 det() const requires(r == c && r == 1)
    {
        return this->dat[0][0];
    }
    constexpr inline f32 det() const requires(r == c && r > 1)
    {
        return Seq<c>::det(*this);
    }

    friend std::ostream& operator<<(std::ostream& s, mat m)
    {
        return Seq<c>::stream(s, m);
    }
};

inline mat4 perspective(f32 fov, f32 ar, f32 n, f32 f)
{
    mat4      m = {};
    const f32 a = 1.f / tanf(fov * 0.5f);
    const f32 d = -1.f / (f - n);
    m[0][0]     = a / ar;
    m[1][1]     = a;
    m[2][2]     = d * (f + n);
    m[2][3]     = -1.f;
    m[3][2]     = d * (2.f * f * n);
    return m;
}

inline mat3 axis_angle(vec3 axis, f32 angle)
{
    axis.normalize();
    f32  c = cosf(angle);
    vec4 s = vec4(sinf(angle) * axis, c);
    return axis.outer((1 - c) * axis) + mat3{s.wzy.sign<0, 0, 1>(),
                                             s.zwx.sign<1, 0, 0>(),
                                             s.yxw.sign<0, 1, 0>()};
}

struct quat : vec_data<4>
{
    constexpr inline quat()
    {
        xyzw = vec4(0, 0, 0, 1);
    }
    constexpr inline quat(vec3 axis)
    {
        xyzw = vec4(axis, 0);
    }

    inline quat(vec3 axis, f32 angle)
    {
        axis.normalize();
        angle *= 0.5f;
        w   = cosf(angle);
        xyz = sinf(angle) * axis;
    }

    explicit constexpr inline quat(vec4 v)
    {
        xyzw = v.xyzw;
    }

    constexpr inline f32 dot(quat q) const
    {
        return xyzw.dot(q.xyzw);
    }

    constexpr inline quat operator*(quat q) const
    {
        return ~quat(zxyw * q.yzxw - wwwx * q.xyzx - xyzy * q.wwwy - yzxz * q.zxyz);
    }

    constexpr inline vec3 operator*(vec3 v) const
    {
        const vec4 s = xyzw * xyzw;
        const vec3 a = s.xyz + s.www - s.yzx - s.zxy;
        const vec3 w = yzx * www;
        const vec3 x = xyz * zxy;
        const vec3 c = v.xyz * (x - w);
        const vec3 d = v.zxy * (x + w);
        return v * a + 2.f * (c.yzx + d);
    }

    constexpr inline quat operator~() const
    {
        return quat(vec4(-xyz, w));
    }
};
