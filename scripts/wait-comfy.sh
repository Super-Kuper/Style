#!/usr/bin/env bash
# =============================================================================
# wait-comfy.sh — ожидание готовности ComfyUI по GET /system_stats.
#
# Используется docker/entrypoint.sh после старта ComfyUI, но пригоден и для
# ручной проверки. ComfyUI доступен только на loopback внутри контейнера
#: по умолчанию http://127.0.0.1:8188.
#
# Использование:
#   scripts/wait-comfy.sh [--url URL] [--timeout SECONDS] [--interval SECONDS]
#                         [--pid PID] [--quiet]
#
#   --url URL            базовый URL ComfyUI (env COMFY_URL, по умолчанию
#                        http://127.0.0.1:8188)
#   --timeout SECONDS    общий таймаут ожидания (env COMFY_BOOT_TIMEOUT_SECONDS,
#                        по умолчанию 1200)
#   --interval SECONDS   пауза между попытками (env COMFY_WAIT_INTERVAL_SECONDS,
#                        по умолчанию 1)
#   --pid PID            если процесс ComfyUI умер — сразу выйти (entrypoint
#                        передаёт сюда pid ComfyUI, чтобы не ждать впустую)
#   --quiet              не печатать прогресс, только итог/ошибку
#   -h, --help           эта справка
#
# Коды выхода (понятные вызывающему, вход в entrypoint учитывает их через if):
#   0 — ComfyUI ответил 200 на /system_stats (готов)
#   1 — таймаут: ComfyUI не стал готов за отведённое время
#   2 — ошибка аргументов
#   3 — нет HTTP-инструмента (curl и python3 отсутствуют)
#   4 — процесс ComfyUI (--pid) завершился, не дождавшись готовности
# =============================================================================

set -euo pipefail

URL="${COMFY_URL:-http://127.0.0.1:8188}"
TIMEOUT="${COMFY_BOOT_TIMEOUT_SECONDS:-1200}"
INTERVAL="${COMFY_WAIT_INTERVAL_SECONDS:-1}"
WATCH_PID=""
QUIET=0

log() { printf '[wait-comfy] %s\n' "$*"; }
log_err() { printf '[wait-comfy] ERROR: %s\n' "$*" >&2; }

usage() {
    cat <<'USAGE'
wait-comfy.sh — ожидание готовности ComfyUI по GET /system_stats.

Использование:
  wait-comfy.sh [--url URL] [--timeout SECONDS] [--interval SECONDS]
                [--pid PID] [--quiet]

  --url URL            базовый URL ComfyUI (env COMFY_URL, по умолчанию
                       http://127.0.0.1:8188)
  --timeout SECONDS    общий таймаут ожидания (env COMFY_BOOT_TIMEOUT_SECONDS,
                       по умолчанию 1200)
  --interval SECONDS   пауза между попытками (env COMFY_WAIT_INTERVAL_SECONDS,
                       по умолчанию 1)
  --pid PID            если процесс ComfyUI умер — сразу выйти (entrypoint
                       передаёт сюда pid ComfyUI, чтобы не ждать впустую)
  --quiet              не печатать прогресс, только итог/ошибку
  -h, --help           эта справка

Коды выхода: 0 — готов; 1 — таймаут; 2 — ошибка аргументов;
             3 — нет curl/python3; 4 — процесс ComfyUI (--pid) умер.
USAGE
}

_is_uint() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --url)
                if [[ $# -lt 2 ]]; then
                    log_err "--url требует значение"
                    exit 2
                fi
                URL="$2"
                shift 2
                ;;
            --timeout)
                if [[ $# -lt 2 ]]; then
                    log_err "--timeout требует значение"
                    exit 2
                fi
                TIMEOUT="$2"
                shift 2
                ;;
            --interval)
                if [[ $# -lt 2 ]]; then
                    log_err "--interval требует значение"
                    exit 2
                fi
                INTERVAL="$2"
                shift 2
                ;;
            --pid)
                if [[ $# -lt 2 ]]; then
                    log_err "--pid требует значение"
                    exit 2
                fi
                WATCH_PID="$2"
                shift 2
                ;;
            --quiet)
                QUIET=1
                shift
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            *)
                log_err "неизвестный аргумент: $1"
                usage >&2
                exit 2
                ;;
        esac
    done

    if ! _is_uint "${TIMEOUT}" || ((TIMEOUT < 1)); then
        log_err "--timeout должен быть целым числом ≥ 1, получено: '${TIMEOUT}'"
        exit 2
    fi
    if ! _is_uint "${INTERVAL}" || ((INTERVAL < 1)); then
        log_err "--interval должен быть целым числом ≥ 1, получено: '${INTERVAL}'"
        exit 2
    fi
    if [[ -n "${WATCH_PID}" ]] && ! _is_uint "${WATCH_PID}"; then
        log_err "--pid должен быть числом, получено: '${WATCH_PID}'"
        exit 2
    fi
    # Нормализуем хвостовой слэш, чтобы не получить //system_stats.
    URL="${URL%/}"
    if [[ -z "${URL}" ]]; then
        log_err "--url не может быть пустым"
        exit 2
    fi
}

# HTTP-проверка: curl есть в образе (Dockerfile); python3 — запасной
# вариант для локальной отладки на машине без curl.
CURL_BIN=""
PY_BIN=""
_detect_http_tool() {
    if command -v curl >/dev/null 2>&1; then
        CURL_BIN="$(command -v curl)"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        PY_BIN="$(command -v python3)"
        return 0
    fi
    return 1
}

_http_ready() {
    if [[ -n "${CURL_BIN}" ]]; then
        # -f: HTTP >= 400 = ошибка; -sS: без прогресс-бара, но с ошибками;
        # --max-time 5: не зависать на полумёртвом ComfyUI.
        "${CURL_BIN}" -fsS --max-time 5 -o /dev/null "${URL}/system_stats"
        return
    fi
    "${PY_BIN}" - "${URL}/system_stats" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        if response.status != 200:
            sys.exit(1)
except Exception:  # noqa: BLE001 — любая сетевая ошибка = "ещё не готов"
    sys.exit(1)
PY
}

main() {
    parse_args "$@"

    if ! _detect_http_tool; then
        log_err "нет ни curl, ни python3 — нечем проверить ${URL}/system_stats"
        exit 3
    fi

    local start="${SECONDS}"
    local attempt=0
    local elapsed=0
    local next_report=0

    while true; do
        attempt=$((attempt + 1))
        if _http_ready >/dev/null 2>&1; then
            elapsed=$((SECONDS - start))
            log "ComfyUI готов: ${URL}/system_stats (${elapsed}s, попыток: ${attempt})"
            exit 0
        fi

        if [[ -n "${WATCH_PID}" ]] && ! kill -0 "${WATCH_PID}" 2>/dev/null; then
            log_err "процесс ComfyUI (pid ${WATCH_PID}) завершился до готовности ${URL}/system_stats (${SECONDS}s, попыток: ${attempt})"
            exit 4
        fi

        elapsed=$((SECONDS - start))
        if ((elapsed >= TIMEOUT)); then
            log_err "ComfyUI не ответил на ${URL}/system_stats за ${TIMEOUT}s (попыток: ${attempt})"
            exit 1
        fi

        if ((QUIET == 0 && elapsed >= next_report)); then
            log "ожидание ComfyUI ${URL}/system_stats: ${elapsed}s/${TIMEOUT}s (попыток: ${attempt})"
            next_report=$((elapsed + 10))
        fi

        sleep "${INTERVAL}"
    done
}

main "$@"
