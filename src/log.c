// Minimal logger implementation for C23 build.

#include <inttypes.h>
#include <stdarg.h>
#include <time.h>

#include "kaldi-native-fbank/log.h"

static knf_log_level g_log_level = KNF_LOG_INFO;

knf_log_level knf_get_log_level() { return g_log_level; }

void knf_set_log_level(knf_log_level level) { g_log_level = level; }

static void knf_vlog(knf_log_level level, const char *file, const char *func,
                     int line, const char *fmt, va_list args) {
  if (level < g_log_level)
    return;

  auto label = "?";
  switch (level) {
  case KNF_LOG_TRACE:
    label = "T";
    break;
  case KNF_LOG_DEBUG:
    label = "D";
    break;
  case KNF_LOG_INFO:
    label = "I";
    break;
  case KNF_LOG_WARNING:
    label = "W";
    break;
  case KNF_LOG_ERROR:
    label = "E";
    break;
  case KNF_LOG_FATAL:
    label = "F";
    break;
  }

  time_t now = time(nullptr);
  struct tm tm_now;
  localtime_r(&now, &tm_now);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_now);

  fprintf(stderr, "[%s] [%s] %s:%d:%s ", buf, label, file, line, func);
  vfprintf(stderr, fmt, args);
  fputc('\n', stderr);
  if (level == KNF_LOG_FATAL) {
    fflush(stderr);
    abort();
  }
}

void knf_log_message(knf_log_level level, const char *file, const char *func,
                     int line, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  knf_vlog(level, file, func, line, fmt, args);
  va_end(args);
}

void knf_fail(const char *expr, const char *file, const char *func, int line,
              const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "Check failed: %s ", expr);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fputc('\n', stderr);
  knf_log_message(KNF_LOG_FATAL, file, func, line,
                  "aborting after failed check");
  abort();
}
