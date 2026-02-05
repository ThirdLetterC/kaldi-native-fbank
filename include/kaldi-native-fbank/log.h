// include/log.h
// Minimal C23 logging and check helpers for KNF.
// This intentionally replaces the old C++ stream-style logger.

#ifndef KALDI_NATIVE_FBANK_CSRC_LOG_H_
#define KALDI_NATIVE_FBANK_CSRC_LOG_H_

#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  KNF_LOG_TRACE = 0,
  KNF_LOG_DEBUG = 1,
  KNF_LOG_INFO = 2,
  KNF_LOG_WARNING = 3,
  KNF_LOG_ERROR = 4,
  KNF_LOG_FATAL = 5
} knf_log_level;

knf_log_level knf_get_log_level();
void knf_set_log_level(knf_log_level level);
void knf_log_message(knf_log_level level, const char *file, const char *func,
                     int line, const char *fmt, ...);
_Noreturn void knf_fail(const char *expr, const char *file, const char *func,
                        int line, const char *fmt, ...);

#define KNF_STATIC_ASSERT(expr) _Static_assert(expr, #expr)

#ifdef KNF_ENABLE_CHECK
#define KNF_CHECK(x)                                                           \
  ((x) ? (void)0 : knf_fail(#x, __FILE__, __func__, __LINE__, ""))
#define KNF_CHECK_EQ(x, y)                                                     \
  ((x) == (y)                                                                  \
       ? (void)0                                                               \
       : knf_fail(#x " == " #y, __FILE__, __func__, __LINE__,                  \
                  "(%" PRId64 " vs %" PRId64 ")", (int64_t)(x), (int64_t)(y)))
#define KNF_CHECK_NE(x, y)                                                     \
  ((x) != (y) ? (void)0                                                        \
              : knf_fail(#x " != " #y, __FILE__, __func__, __LINE__, ""))
#define KNF_CHECK_LT(x, y)                                                     \
  ((x) < (y) ? (void)0                                                         \
             : knf_fail(#x " < " #y, __FILE__, __func__, __LINE__, ""))
#define KNF_CHECK_LE(x, y)                                                     \
  ((x) <= (y) ? (void)0                                                        \
              : knf_fail(#x " <= " #y, __FILE__, __func__, __LINE__, ""))
#define KNF_CHECK_GT(x, y)                                                     \
  ((x) > (y) ? (void)0                                                         \
             : knf_fail(#x " > " #y, __FILE__, __func__, __LINE__, ""))
#define KNF_CHECK_GE(x, y)                                                     \
  ((x) >= (y) ? (void)0                                                        \
              : knf_fail(#x " >= " #y, __FILE__, __func__, __LINE__, ""))
#else
#define KNF_CHECK(x) ((void)0)
#define KNF_CHECK_EQ(x, y) ((void)0)
#define KNF_CHECK_NE(x, y) ((void)0)
#define KNF_CHECK_LT(x, y) ((void)0)
#define KNF_CHECK_LE(x, y) ((void)0)
#define KNF_CHECK_GT(x, y) ((void)0)
#define KNF_CHECK_GE(x, y) ((void)0)
#endif

#ifdef KNF_ENABLE_CHECK
#define KNF_DCHECK(x) KNF_CHECK(x)
#define KNF_DCHECK_EQ(x, y) KNF_CHECK_EQ(x, y)
#define KNF_DCHECK_NE(x, y) KNF_CHECK_NE(x, y)
#define KNF_DCHECK_LT(x, y) KNF_CHECK_LT(x, y)
#define KNF_DCHECK_LE(x, y) KNF_CHECK_LE(x, y)
#define KNF_DCHECK_GT(x, y) KNF_CHECK_GT(x, y)
#define KNF_DCHECK_GE(x, y) KNF_CHECK_GE(x, y)
#else
#define KNF_DCHECK(x) ((void)0)
#define KNF_DCHECK_EQ(x, y) ((void)0)
#define KNF_DCHECK_NE(x, y) ((void)0)
#define KNF_DCHECK_LT(x, y) ((void)0)
#define KNF_DCHECK_LE(x, y) ((void)0)
#define KNF_DCHECK_GT(x, y) ((void)0)
#define KNF_DCHECK_GE(x, y) ((void)0)
#endif

#define KNF_LOG(level, fmt, ...)                                               \
  knf_log_message(level, __FILE__, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define KNF_LOG_TRACE(fmt, ...) KNF_LOG(KNF_LOG_TRACE, fmt, ##__VA_ARGS__)
#define KNF_LOG_DEBUG(fmt, ...) KNF_LOG(KNF_LOG_DEBUG, fmt, ##__VA_ARGS__)
#define KNF_LOG_INFO(fmt, ...) KNF_LOG(KNF_LOG_INFO, fmt, ##__VA_ARGS__)
#define KNF_LOG_WARN(fmt, ...) KNF_LOG(KNF_LOG_WARNING, fmt, ##__VA_ARGS__)
#define KNF_LOG_ERROR(fmt, ...) KNF_LOG(KNF_LOG_ERROR, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // KALDI_NATIVE_FBANK_CSRC_LOG_H_
