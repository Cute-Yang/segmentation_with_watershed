#pragma once

#define FISHAUX_CONCAT_EXP(a, b) a##b
#define FISHAUX_CONCAT(a, b) FISHAUX_CONCAT_EXP(a, b)
#define FISHAPI_EXPORTS 1

// define the assert macro!
#if defined(__clang__)
#    ifndef __has_extension
#        define __has_extension __has_feature /* compatibility, for older versions of clang */
#    endif
#    if __has_extension(cxx_static_assert)
#        define FISH_StaticAssert(condition, reason) \
            static_assert((condition), reason " " #condition)
#    elif __has_extension(c_static_assert)
#        define FISH_StaticAssert(condition, reason) \
            _Static_assert((condition), reason " " #condition)
#    endif
#elif defined(__GNUC__)
#    if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L)
#        define FISH_StaticAssert(condition, reason) \
            static_assert((condition), reason " " #condition)
#    endif
#elif defined(_MSC_VER)
#    if _MSC_VER >= 1600 /* MSVC 10 */
#        define FISH_StaticAssert(condition, reason) \
            static_assert((condition), reason " " #condition)
#    endif
#endif


#ifndef FISH_StaticAssert
#    if !defined(__clang__) && defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ > 302)
#        define FISH_StaticAssert(condition, reason)                                           \
            ({                                                                                 \
                extern int __attribute__((error("FISH_StaticAssert: " reason " " #condition))) \
                FISH_StaticAssert();                                                           \
                ((condition) ? 0 : FISH_StaticAssert());                                       \
            })
#    else
namespace fish {
template<bool x> struct FISH_StaticAssert_failed;
template<> struct FISH_StaticAssert_failed<true> {
    enum { val = 1 };
};
template<int x> struct FISH_StaticAssert_test {};
}   // namespace fish
#        define FISH_StaticAssert(condition, reason)                           \
            typedef FISH::FISH_StaticAssert_test<sizeof(                       \
                fish::FISH_StaticAssert_failed<static_cast<bool>(condition)>)> \
            FISHAUX_CONCAT(FISH_StaticAssert_failed_at_, __LINE__)
#    endif
#endif


// define inline
#ifndef FISH_INLINE
#    if defined __cplusplus
#        define FISH_INLINE static inline
#    elif defined _MSC_VER
#        define FISH_INLINE __inline
#    else
#        define FISH_INLINE static
#    endif
#endif

// define always inline!
#ifndef FISH_ALWAYS_INLINE
#    if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#        define FISH_ALWAYS_INLINE inline __attribute__((always_inline))
#    elif defined(_MSC_VER)
#        define FISH_ALWAYS_INLINE __forceinline
#    else
#        define FISH_ALWAYS_INLINE inline
#    endif
#endif

// define msvc export
#ifndef FISH_EXPORTS
#    if (defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined(SegAPI_EXPORTS)
#        define FISH_EXPORTS __declspec(dllexport)
#    elif defined __GNUC__ && __GNUC__ >= 4 && (defined(FISHAPI_EXPORTS) || defined(__APPLE__))
#        define FISH_EXPORTS __attribute__((visibility("default")))
#    else
#        define FISH_EXPORTS
#    endif
#endif

#define FISH_EXPORTS_W FISH_EXPORTS

#ifndef FISH_DEPRECATED
#    if defined(__GNUC__)
#        define FISH_DEPRECATED __attribute__((deprecated))
#    elif defined(_MSC_VER)
#        define FISH_DEPRECATED __declspec(deprecated)
#    else
#        define FISH_DEPRECATED
#    endif
#endif

// define export C,generate abi with c style!
#ifndef FISH_EXTERN_C
#    ifdef __cplusplus
#        define FISH_EXTERN_C extern "C"
#    else
#        define FISH_EXTERN_C
#    endif
#endif

// define simple min/max macro!
#ifndef FISH_MIN
#    define FISH_MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

#ifndef FISH_MAX
#    define FISH_MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

#ifndef FISH_CLIP
#    define FISH_CLIP(x, a, b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#endif


// support nodiscard!
#ifndef FISH_NODISCARD_STD
#    ifndef __has_cpp_attribute
//   workaround preprocessor non-compliance https://reviews.llvm.org/D57851
#        define __has_cpp_attribute(__x) 0
#    endif
#    if __has_cpp_attribute(nodiscard)
#        define FISH_NODISCARD_STD [[nodiscard]]
#    elif __cplusplus >= 201703L
    //   available when compiler is C++17 compliant
#        define FISH_NODISCARD_STD [[nodiscard]]
#    elif defined(__INTEL_COMPILER)
// see above, available when C++17 is enabled
#    elif defined(_MSC_VER) && _MSC_VER >= 1911 && _MSVC_LANG >= 201703L
//   available with VS2017 v15.3+ with /std:c++17 or higher; works on functions and classes
#        define FISH_NODISCARD_STD [[nodiscard]]
#    elif defined(__GNUC__) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 700) && \
        (__cplusplus >= 201103L)
//   available with GCC 7.0+; works on functions, works or silently fails on classes
#        define FISH_NODISCARD_STD [[nodiscard]]
#    elif defined(__GNUC__) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 408) && \
        (__cplusplus >= 201103L)
//   available with GCC 4.8+ but it usually does nothing and can fail noisily -- therefore not used
//   define FISH_NODISCARD_STD [[gnu::warn_unused_result]]
#    endif
#endif
#ifndef FISH_NODISCARD_STD
#    define FISH_NODISCARD_STD /* nothing by default */
#endif


#ifndef FISH_NODISCARD
#    if defined(__GNUC__)
#        define FISH_NODISCARD __attribute__((__warn_unused_result__))
#    elif defined(__clang__) && defined(__has_attribute)
#        if __has_attribute(__warn_unused_result__)
#            define FISH_NODISCARD __attribute__((__warn_unused_result__))
#        endif
#    endif
#endif
#ifndef FISH_NODISCARD
#    define FISH_NODISCARD /* nothing by default */
#endif

#ifndef FISH_LIKELY_STD
#    ifndef __has_cpp_attribute
#        define __has_cpp_attribute(__x) 0
#    endif
#    if __has_cpp_attribute(likely)
#        define FISH_LIKELY_STD [[likely]]
#    elif __cplusplus >= 202002L
    //   available when compiler is C++20 compliant
#        define FISH_LIKELY_STD [[likely]]
#    elif defined(__INTEL_COMPILER)
// see above, available when C++17 is enabled
#    elif defined(_MSC_VER) && _MSC_VER >= 1911 && _MSVC_LANG >= 202002L
//   available with VS2017 v15.3+ with /std:c++17 or higher; works on functions and classes
#        define FISH_LIKELY_STD [[likely]]
#    else
// define nothing!
#        define FISH_LIKELY_STD
//   available with GCC 4.8+ but it usually does nothing and can fail noisily -- therefore not used
//   define FISH_NODISCARD_STD [[gnu::warn_unused_result]]
#    endif
#endif


#ifndef FISH_UNLIKELY_STD
#    ifndef __has_cpp_attribute
#        define __has_cpp_attribute(__x) 0
#    endif
#    if __has_cpp_attribute(unlikely)
#        define FISH_UNLIKELY_STD [[unlikely]]
#    elif __cplusplus >= 202002L
    //   available when compiler is C++20 compliant
#        define FISH_UNLIKELY_STD [[unlikely]]
#    elif defined(__INTEL_COMPILER)
// see above, available when C++20 is enabled
#    elif defined(_MSC_VER) && _MSC_VER >= 1911 && _MSVC_LANG >= 202002L
//   available with VS2017 v15.3+ with /std:c++17 or higher; works on functions and classes
#        define FISH_UNLIKELY_STD [[unlikely]]
#    else
// define nothing!
#        define FISH_UNLIKELY_STD
//   available with GCC 4.8+ but it usually does nothing and can fail noisily -- therefore not used
//   define FISH_NODISCARD_STD [[gnu::warn_unused_result]]
#    endif
#endif


// float16 support!
#if !defined _MSC_VER && !defined __BORLANDC__
#    if defined __cplusplus && __cplusplus >= 201103L && !defined __APPLE__
#        include <cstdint>
#        ifdef __NEWLIB__
typedef unsigned int uint;
#        else
typedef std::uint32_t uint;
#        endif
#    else
#        include <stdint.h>
typedef uint32_t uint;
#    endif
#else
typedef unsigned uint;
#endif

typedef signed char schar;

#ifndef __IPL_H__
typedef unsigned char  uchar;
typedef unsigned short ushort;
#endif

#if defined _MSC_VER || defined __BORLANDC__
typedef __int64                 int64;
typedef unsigned __int64        uint64;
#    define FISH_BIG_INT(n) n##I64
#    define FISH_BIG_UINT(n) n##UI64
#else
typedef int64_t  int64;
typedef uint64_t uint64;
#    define FISH_BIG_INT(n) n##LL
#    define FISH_BIG_UINT(n) n##ULL
#endif


#if defined __ARM_FP16_FORMAT_IEEE && !defined __CUDACC__
#    define FISH_FP16_TYPE 1
#else
#    define FISH_FP16_TYPE 0
#endif

typedef union FISH16suf {
    short  i;
    ushort u;
#if FISH_FP16_TYPE
    __fp16 h;
#endif
} FISH16suf;

typedef union FISH32suf {
    int      i;
    unsigned u;
    float    f;
} FISH32suf;

typedef union FISH64suf {
    int64  i;
    uint64 u;
    double f;
} FISH64suf;

#ifdef __cplusplus
namespace fish {
class float16_t {
public:
#    if FISH_FP16_TYPE
    float16_t()
        : h(0) {}
    explicit float16_t(float x) { h = (__fp16)x; }
                     operator float() const { return (float)h; }
    static float16_t fromBits(ushort w) {
        Cv16suf u;
        u.u = w;
        float16_t result;
        result.h = u.h;
        return result;
    }
    static float16_t zero() {
        float16_t result;
        result.h = (__fp16)0;
        return result;
    }
    ushort bits() const {
        Cv16suf u;
        u.h = h;
        return u.u;
    }

protected:
    __fp16 h;

#    else
    float16_t()
        : w(0) {}
    explicit float16_t(float x) {
#        if FISH_FP16
        __m128 v = _mm_load_ss(&x);
        w        = (ushort)_mm_cvtsi128_si32(_mm_cvtps_ph(v, 0));
#        else
        FISH32suf in;
        in.f          = x;
        unsigned sign = in.u & 0x80000000;
        in.u ^= sign;

        if (in.u >= 0x47800000)
            w = (ushort)(in.u > 0x7f800000 ? 0x7e00 : 0x7c00);
        else {
            if (in.u < 0x38800000) {
                in.f += 0.5f;
                w = (ushort)(in.u - 0x3f000000);
            } else {
                unsigned t = in.u + 0xc8000fff;
                w          = (ushort)((t + ((in.u >> 13) & 1)) >> 13);
            }
        }

        w = (ushort)(w | (sign >> 16));
#        endif
    }

    operator float() const {
#        if FISH_FP16
        float f;
        _mm_store_ss(&f, _mm_cvtph_ps(_mm_cvtsi32_si128(w)));
        return f;
#        else
        FISH32suf out;

        unsigned t    = ((w & 0x7fff) << 13) + 0x38000000;
        unsigned sign = (w & 0x8000) << 16;
        unsigned e    = w & 0x7c00;

        out.u = t + (1 << 23);
        out.u = (e >= 0x7c00 ? t + 0x38000000
                 : e == 0    ? (static_cast<void>(out.f -= 6.103515625e-05f), out.u)
                             : t) |
                sign;
        return out.f;
#        endif
    }

    static float16_t fromBits(ushort b) {
        float16_t result;
        result.w = b;
        return result;
    }
    static float16_t zero() {
        float16_t result;
        result.w = (ushort)0;
        return result;
    }
    ushort bits() const { return w; }

protected:
    ushort w;

#    endif
};

}   // namespace fish
#endif



#ifdef FISH_STDINT_HEADER
#    include FISH_STDINT_HEADER
#elif defined(__cplusplus)
#    if defined(_MSC_VER) && _MSC_VER < 1600 /* MSVS 2010 */
namespace fish {
typedef signed char      int8_t;
typedef unsigned char    uint8_t;
typedef signed short     int16_t;
typedef unsigned short   uint16_t;
typedef signed int       int32_t;
typedef unsigned int     uint32_t;
typedef signed __int64   int64_t;
typedef unsigned __int64 uint64_t;
}   // namespace fish
#    elif defined(_MSC_VER) || __cplusplus >= 201103L
#        include <cstdint>
namespace fish {
using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;
}   // namespace fish
#    else
#        include <stdint.h>
namespace fish {
typedef ::int8_t   int8_t;
typedef ::uint8_t  uint8_t;
typedef ::int16_t  int16_t;
typedef ::uint16_t uint16_t;
typedef ::int32_t  int32_t;
typedef ::uint32_t uint32_t;
typedef ::int64_t  int64_t;
typedef ::uint64_t uint64_t;
}   // namespace fish
#    endif
#else   // pure C
#    include <stdint.h>
#endif


// func name
#ifdef FISH_FUNC
// keep current value (through OpenCV port file)
#elif defined __GNUC__ || (defined(__cpluscplus) && (__cpluscplus >= 201103))
#    define FISH_FUNC __func__
#elif defined __clang__ && (__clang_minor__ * 100 + __clang_major__ >= 305)
#    define FISH_FUNC __func__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION >= 199901)
#    define FISH_FUNC __func__
#elif defined _MSC_VER
#    define FISH_FUNC __FUNCTION__
#elif defined(__INTEL_COMPILER) && (_INTEL_COMPILER >= 600)
#    define FISH_FUNC __FUNCTION__
#elif defined __IBMCPP__ && __IBMCPP__ >= 500
#    define FISH_FUNC __FUNCTION__
#elif defined __BORLAND__ && (__BORLANDC__ >= 0x550)
#    define FISH_FUNC __FUNC__
#else
#    define FISH_FUNC "<lazydog>"
#endif

#ifndef FISH_OVERRIDE
#    define FISH_OVERRIDE override
#endif
#ifndef FISH_FINAL
#    define FISH_FINAL final
#endif
