#!/usr/bin/env bash
# =============================================================================
# populate-volume.sh — модели на сетевом томе RunPod: докачка и сверка.
#
# Читает models-manifest.txt (tab-separated: <subdir> <filename> <URL> [<sha256>])
# и кладёт файлы в ${VOL_ROOT}/models/<subdir>/<filename> — это тот же layout,
# который extra_model_paths.yaml отдаёт ComfyUI. Скрипт идемпотентен:
#   - уже существующий файл проверяется по pinned sha256 (или по хешу из
#     состояния тома ${MODELS_DIR}/.populate-state.tsv) и пропускается;
#   - расхождение/битый файл отводится в карантин <file>.corrupt-<UTC> и
#     скачивается заново (проверенные байты сохраняются для разбора);
#   - докачка идёт в <dest>.tmp с `curl --continue-at -`, затем атомарный
#     `mv -f` в целевое имя (повторный job/boot не видит полуфайл);
#   - конкурентные воркеры сериализуются flock'ом на ${VOL_ROOT}/.populate.lock,
#     но ТОЛЬКО вокруг изменений (скачивание/карантин/запись состояния), а не
#     вокруг чтения: фоновая сверка одного воркера не блокирует старт другого.
#
# Пины обязательны: URL без commit-ревизии (`/resolve/<40-hex>/`) отвергается,
# `main`/`master` запрещены. Если манифест использует placeholder `${VAR}`, он
# раскрывается из окружения, а нераскрытый placeholder — ошибка (лучше упасть с
# понятным текстом, чем молча скачать `main`). Pinned sha256 из 4-й колонки
# проверяется всегда, если задан.
#
# Режимы сверки (VERIFY):
#   fast  — режим по умолчанию. Файл считается целым, если в состоянии тома
#           совпадают sha256, размер и mtime; sha256 читается только когда
#           метаданные разошлись (новый пин, чужой файл, правка на томе) или
#           для строк старого формата. На неизменном томе старт занимает
#           секунды вместо ~4.5 минут чтения 66.7 GB.
#   full  — sha256 читается всегда. Именно этот режим запускает entrypoint
#           фоном после старта handler'а: полная сверка остаётся на каждом
#           boot, но больше не задерживает воркер.
#   off   — проверяется только наличие файла (отладка). Битые веса будут
#           молча использованы: в рабочем endpoint'е так делать нельзя.
#
# Состояние тома ${MODELS_DIR}/.populate-state.tsv:
#   <subdir>/<filename> \t sha256 \t размер \t mtime
#   Строки старого формата (без mtime) читаются как «метаданные неизвестны»:
#   файл сверяется по sha256 и получает mtime при следующей успешной проверке,
#   поэтому переход на новый формат не требует перекачки и не ломает старые тома.
#
# Env:
#   VOL_ROOT         точка монтирования тома    (по умолчанию /runpod-volume)
#   MANIFEST         путь к манифесту моделей   (по умолчанию рядом со скриптом)
#   VERIFY           fast|full|off              (по умолчанию fast)
#   POPULATE_LABEL   префикс лога               (по умолчанию populate;
#                    entrypoint ставит populate-verify для фоновой сверки)
#   HF_TOKEN         токен Hugging Face для gated-репозиториев (опционально)
#   HF_ENDPOINT      зеркало Hugging Face       (по умолчанию https://huggingface.co)
#
# Вызывается из docker/entrypoint.sh: при VOLUME_AUTOPOPULATE=true — в
# фореграунде с VERIFY=fast (докачка до старта handler'а), затем повторно
# фоном с VERIFY=full. При ошибке скачивания entrypoint продолжает старт, а
# job'ы падают с видимой ошибкой отсутствующих моделей (а не «тихо» работают
# на других весах).
# =============================================================================

set -uo pipefail

VOL_ROOT="${VOL_ROOT:-/runpod-volume}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-${SCRIPT_DIR}/models-manifest.txt}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
VERIFY="${VERIFY:-fast}"
LABEL="${POPULATE_LABEL:-populate}"
MODELS_DIR="${VOL_ROOT}/models"
STATE_FILE="${MODELS_DIR}/.populate-state.tsv"
LOCK_FILE="${VOL_ROOT}/.populate.lock"

case "${VERIFY}" in
    fast|full|off) ;;
    *)
        printf '[%s] ERROR: VERIFY=%s; ожидалось fast|full|off\n' "${LABEL}" "${VERIFY}" >&2
        exit 1
        ;;
esac

log() { printf '[%s] %s\n' "${LABEL}" "$*"; }
warn() { printf '[%s] WARN: %s\n' "${LABEL}" "$*" >&2; }
die() {
    printf '[%s] ERROR: %s\n' "${LABEL}" "$*" >&2
    exit 1
}

if [[ ! -d "${VOL_ROOT}" ]]; then
    log "том ${VOL_ROOT} не смонтирован — скачивание пропущено (attach network volume к endpoint'у)"
    exit 0
fi
if [[ ! -f "${MANIFEST}" ]]; then
    die "нет манифеста ${MANIFEST}: образ собран некорректно (ожидался COPY models-manifest.txt)"
fi

# --- Инструменты -------------------------------------------------------------

#: Команда хеша — массив, чтобы "shasum -a 256" не разбирался на два слова.
SHA_CMD=()
if command -v sha256sum >/dev/null 2>&1; then
    SHA_CMD=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
    SHA_CMD=(shasum -a 256)
fi

_sha256() {
    # Печатает хеш; ненулевой код — сбой инструмента/ФС (например, ERANGE на
    # большом файле сетевого тома). Вызывающий обязан отличать это от
    # расхождения хешей: сбой инструмента не повод удалять скачанный файл.
    local out
    out="$("${SHA_CMD[@]}" -- "$1")" || return 1
    printf '%s\n' "${out%% *}"
}

# Размер и mtime (epoch, секунды). GNU stat первый, BSD — запасной вариант.
_file_size() {
    stat -c %s -- "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0
}

_file_mtime() {
    stat -c %Y -- "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}

_human() {
    awk -v b="$1" 'BEGIN { if (b ~ /^[0-9]+$/) printf "%.2f GB", b / 1e9; else print "?" }'
}

# --- Раскрытие ${VAR} в URL ---------------------------------------------------
# Разрешаем только имена [A-Z0-9_]+ и подставляем значения литерально; eval не
# используется (URL — внешние данные, shell-метасимволы в них не исполняем).
_expand_url() {
    local url="$1" name value
    while IFS= read -r name; do
        [[ -n "${name}" ]] || continue
        value="${!name-}"
        url="${url//\$\{${name}\}/${value}}"
    done < <(printf '%s' "${url}" | grep -oE '\$\{[A-Z0-9_]+\}' | sed -e 's/^\${//' -e 's/}$//' | sort -u)
    printf '%s\n' "${url}"
}

# --- Проверка пина/формата ----------------------------------------------------

_validate_url() {
    local url="$1" where="$2"

    if [[ -z "${url}" ]]; then
        die "${where}: пустой URL"
    fi
    if [[ "${url}" == *'${'* || "${url}" == *'$'* ]]; then
        die "${where}: нераскрытый placeholder в URL (${url}). Задайте переменную ревизии/пути в env — main и «плавающие» ревизии запрещены"
    fi
    if [[ "${url}" =~ /resolve/(main|master)/ ]]; then
        die "${where}: URL указывает на main/master (${url}). Нужен commit-pinned URL: /resolve/<40-hex-commit>/..."
    fi
    if [[ ! "${url}" =~ /resolve/[0-9a-f]{40}/ ]]; then
        die "${where}: URL не запинен на commit (${url}). Ожидался /resolve/<40-hex-commit>/...; pinned-ревизии обязательны"
    fi
}

_validate_sha() {
    local sha="$1" where="$2"
    if [[ -z "${sha}" ]]; then
        warn "${where}: sha256 не задан — проверка будет только по состоянию тома (.populate-state.tsv)"
        return 0
    fi
    if [[ ! "${sha}" =~ ^[0-9a-f]{64}$ ]]; then
        die "${where}: sha256 должен быть 64 lowercase hex, получено '${sha}'"
    fi
}

# --- Состояние тома (sha256/размер/mtime скачанных файлов) --------------------

_state_row() {
    # Печатает "<sha256>\t<размер>\t<mtime>" для ключа (короткий формат —
    # строка старого образца без mtime, её нельзя использовать для fast-path).
    local key="$1"
    [[ -f "${STATE_FILE}" ]] || return 0
    awk -F'\t' -v key="${key}" '$1 == key { sub(/^[^\t]*\t?/, ""); print; exit }' "${STATE_FILE}"
}

_state_set() {
    local key="$1" sha="$2" size="$3" mtime="$4"
    local tmp="${STATE_FILE}.tmp.$$"
    {
        [[ -f "${STATE_FILE}" ]] && awk -F'\t' -v key="${key}" '$1 != key' "${STATE_FILE}"
        printf '%s\t%s\t%s\t%s\n' "${key}" "${sha}" "${size}" "${mtime}"
    } >"${tmp}" && mv -f "${tmp}" "${STATE_FILE}"
}

# --- Блокировка тома ----------------------------------------------------------
# flock берётся только вокруг изменений: чтение/хеширование идёт без блокировки,
# иначе фоновая сверка (минуты на 66.7 GB) задерживала бы старт второго воркера.
# Мутация под блокировкой всегда перепроверяет метаданные файла: если их успел
# изменить другой воркер, действие пропускается.
HAS_FLOCK=0
_lock() {
    [[ "${HAS_FLOCK}" -eq 1 ]] && flock -x 9
}
_unlock() {
    [[ "${HAS_FLOCK}" -eq 1 ]] && flock -u 9
}

# --- Старт -------------------------------------------------------------------

if [[ ${#SHA_CMD[@]} -eq 0 ]]; then
    warn "нет sha256sum/shasum — проверка sha256 невозможна; pinned sha256 в манифесте обязателен, проверьте образ"
fi
if [[ "${VERIFY}" == "off" ]]; then
    warn "VERIFY=off: целостность существующих моделей не проверяется — только отладка, не для рабочего endpoint'а"
fi

started_at="$(date +%s)"
total="$(grep -vcE '^[[:space:]]*(#|$)' "${MANIFEST}" || true)"
log "манифест: ${total} файлов; сверка: ${VERIFY}; свободно на ${VOL_ROOT}: $(df -h "${VOL_ROOT}" 2>/dev/null | awk 'NR==2{print $4}')"

# Сериализация изменений на общем томе (9 — произвольный fd).
if command -v flock >/dev/null 2>&1; then
    exec 9>"${LOCK_FILE}"
    HAS_FLOCK=1
else
    warn "flock не найден — параллельные изменения на томе не сериализуются"
fi

# Layout тома: models/{...}, custom_nodes, encoder-cache создаются заранее.
_lock
mkdir -p "${MODELS_DIR}" "${VOL_ROOT}/custom_nodes" "${VOL_ROOT}/encoder-cache"
if [[ ! -f "${STATE_FILE}" ]]; then
    printf '# subdir/filename\tsha256\tsize\tmtime\n' >"${STATE_FILE}"
fi
_unlock

done_n=0
failed=0
skipped_fast=0
skipped_sha=0
skipped_nosha=0
skipped_off=0
downloaded=0

while IFS=$'\t' read -r sub name url sha extra; do
    sub="${sub%$'\r'}"
    name="${name%$'\r'}"
    url="${url%$'\r'}"
    sha="${sha%$'\r'}"
    extra="${extra%$'\r'}"
    case "${sub}" in ''|\#*) continue ;; esac
    done_n=$((done_n + 1))
    where="строка ${done_n} (${sub}/${name})"

    if [[ -n "${extra}" ]]; then
        die "${where}: лишняя колонка '${extra}'; формат <subdir>\t<filename>\t<URL>[\t<sha256>]"
    fi
    if [[ -z "${name}" || -z "${url}" ]]; then
        warn "${where}: ожидались <subdir> <filename> <URL> [<sha256>]"
        failed=$((failed + 1))
        continue
    fi

    unresolved="$(printf '%s' "${url}" | grep -oE '\$\{[A-Z0-9_]+\}' | sed -e 's/^\${//' -e 's/}$//' | sort -u || true)"
    if [[ -n "${unresolved}" ]]; then
        missing=()
        while IFS= read -r var; do
            [[ -n "${!var-}" ]] || missing+=("${var}")
        done <<<"${unresolved}"
        if [[ ${#missing[@]} -gt 0 ]]; then
            die "${where}: не заданы переменные ревизии/пути: ${missing[*]}. Задайте их в env endpoint'а; main и «плавающие» ревизии запрещены"
        fi
    fi
    url="$(_expand_url "${url}")"
    _validate_url "${url}" "${where}"
    _validate_sha "${sha}" "${where}"

    dest="${MODELS_DIR}/${sub}/${name}"
    key="${sub}/${name}"
    quarantine=""

    # Зеркало Hugging Face: подменяем только префикс, путь/commit-ревизия
    # сохраняются — пин не теряется.
    if [[ "${HF_ENDPOINT}" != "https://huggingface.co" && "${url}" == https://huggingface.co/* ]]; then
        url="${HF_ENDPOINT%/}/${url#https://huggingface.co/}"
    fi

    mkdir -p "${MODELS_DIR}/${sub}"

    # Ожидаемый хеш нужен и до проверки файла: по нему повторная сверка под
    # блокировкой понимает, что второй воркер уже скачал нужный файл.
    expected="${sha}"
    if [[ -z "${expected}" ]]; then
        expected="$(_state_row "${key}" | cut -f1)"
    fi

    if [[ -s "${dest}" ]]; then
        size="$(_file_size "${dest}")"
        mtime="$(_file_mtime "${dest}")"

        if [[ "${VERIFY}" == "off" ]]; then
            log "[${done_n}/${total}] есть ${key} (VERIFY=off: sha не читаю)"
            skipped_off=$((skipped_off + 1))
            continue
        fi

        # fast-path: доверяем паре (размер, mtime) из состояния тома, но только
        # если там записан ровно ожидаемый пин. Смена пина/размера/mtime —
        # повод прочитать файл целиком.
        if [[ "${VERIFY}" == "fast" && -n "${expected}" && ${#SHA_CMD[@]} -gt 0 ]]; then
            IFS=$'\t' read -r state_sha state_size state_mtime <<<"$(_state_row "${key}")" || true
            if [[ -n "${state_sha:-}" && "${state_sha}" == "${expected}" \
                && -n "${state_size:-}" && "${state_size}" == "${size}" \
                && -n "${state_mtime:-}" && "${state_mtime}" == "${mtime}" ]]; then
                log "[${done_n}/${total}] есть ${key} (метаданные совпадают: ${size} байт, mtime ${mtime}; sha не читаю)"
                skipped_fast=$((skipped_fast + 1))
                continue
            fi
        fi

        if [[ ${#SHA_CMD[@]} -eq 0 ]]; then
            log "[${done_n}/${total}] есть ${key} (нет sha256sum/shasum — без проверки)"
            skipped_nosha=$((skipped_nosha + 1))
            continue
        fi

        log "[${done_n}/${total}] проверяю sha256 ${key} ($(_human "${size}"))"
        if ! actual="$(_sha256 "${dest}")"; then
            warn "[${done_n}/${total}] ${key}: sha256 не посчитан (сбой инструмента/ФС) — файл оставляю без проверки"
            failed=$((failed + 1))
            continue
        fi

        if [[ -n "${expected}" && "${actual}" != "${expected}" ]]; then
            # Карантин вместо rm -f: байты сохраняются для разбора, но модель
            # сразу перестаёт быть видимой ComfyUI (job упадёт видимо, а не
            # молча сгенерирует на битых весах) — этого требует правило
            # «отказ от молчаливого фолбэка».
            _lock
            # Пока считался хеш, файл мог заменить другой воркер (его докачка
            # проверена по пину). Тогда карантин не наша забота.
            if [[ "$(_file_size "${dest}")" != "${size}" || "$(_file_mtime "${dest}")" != "${mtime}" ]]; then
                _unlock
                log "[${done_n}/${total}] ${key}: файл уже заменён другим воркером — пропускаю"
                skipped_fast=$((skipped_fast + 1))
                continue
            fi
            quarantine="${dest}.corrupt-$(date -u +%Y%m%dT%H%M%SZ)"
            warn "[${done_n}/${total}] ${key}: sha256 ${actual} != ожидаемого ${expected} — отвожу в карантин и качаю заново"
            if ! mv -f "${dest}" "${quarantine}"; then
                _unlock
                warn "[${done_n}/${total}] ${key}: не удалось отвести файл в карантин (${quarantine})"
                failed=$((failed + 1))
                continue
            fi
            log "[${done_n}/${total}] ${key}: карантин ${quarantine}"
            # Блокировку не снимаем: докачка ниже идёт под ней же.
        else
            _lock
            # Перечитываем метаданные под блокировкой: файл мог быть заменён,
            # пока считался хеш.
            new_size="$(_file_size "${dest}")"
            new_mtime="$(_file_mtime "${dest}")"
            if [[ "${new_size}" == "${size}" && "${new_mtime}" == "${mtime}" ]]; then
                if [[ -z "${expected}" ]]; then
                    _state_set "${key}" "${actual}" "${size}" "${mtime}"
                    log "[${done_n}/${total}] есть ${key}: sha256 ${actual} зафиксирован в состоянии тома"
                else
                    # Обновляем строку состояния (в т.ч. добавляем mtime строкам
                    # старого формата, чтобы следующий boot шёл по fast-path).
                    _state_set "${key}" "${actual}" "${size}" "${mtime}"
                    log "[${done_n}/${total}] есть ${key} (sha256 ok)"
                fi
                skipped_sha=$((skipped_sha + 1))
            else
                log "[${done_n}/${total}] ${key}: файл изменился во время проверки — пропускаю до следующего boot"
                skipped_fast=$((skipped_fast + 1))
            fi
            _unlock
            continue
        fi
    fi

    # --- Докачка --------------------------------------------------------------
    # Под блокировкой ещё раз убеждаемся, что файл не появился, пока мы
    # хешировали/ждали: второй воркер мог уже всё скачать.
    _lock
    if [[ -s "${dest}" && -n "${expected:-}" && ${#SHA_CMD[@]} -gt 0 ]]; then
        IFS=$'\t' read -r state_sha state_size state_mtime <<<"$(_state_row "${key}")" || true
        if [[ -n "${state_sha:-}" && "${state_sha}" == "${expected}" \
            && -n "${state_size:-}" && "${state_size}" == "$(_file_size "${dest}")" \
            && -n "${state_mtime:-}" && "${state_mtime}" == "$(_file_mtime "${dest}")" ]]; then
            _unlock
            log "[${done_n}/${total}] ${key}: появился, пока ждал блокировку — пропускаю"
            skipped_fast=$((skipped_fast + 1))
            continue
        fi
    fi

    log "[${done_n}/${total}] качаю ${key} ← ${url%%\?*}"
    hdr=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        hdr=(--header "Authorization: Bearer ${HF_TOKEN}")
    fi

    # --continue-at -: докачиваем .tmp после обрыва, не начиная файл заново.
    # curl вызывается без `if !`: внутри `if ! cmd; then` значение $? — это
    # результат отрицания (0), и в лог попадал бы «curl exit 0» вместо кода.
    curl --fail --location --silent --show-error \
        --retry 5 --retry-delay 10 --continue-at - \
        "${hdr[@]}" --output "${dest}.tmp" "${url}"
    code=$?
    if [[ ${code} -ne 0 ]]; then
        _unlock
        warn "${where}: не удалось скачать (curl exit ${code}); частичный файл ${dest}.tmp оставлен для докачки"
        failed=$((failed + 1))
        continue
    fi

    if [[ -n "${sha}" && ${#SHA_CMD[@]} -gt 0 ]]; then
        if ! actual="$(_sha256 "${dest}.tmp")"; then
            warn "${where}: sha256 не посчитан (сбой инструмента/ФС) — .tmp оставлен для повторной проверки, а не удалён"
            _unlock
            failed=$((failed + 1))
            continue
        fi
        if [[ "${actual}" != "${sha}" ]]; then
            warn "${where}: sha256 ${actual} != pinned ${sha} — файл повреждён/подменён, удаляю .tmp"
            rm -f "${dest}.tmp"
            _unlock
            failed=$((failed + 1))
            continue
        fi
    fi

    size="$(_file_size "${dest}.tmp")"
    # Атомарная замена в пределах той же ФС: полуфайл никогда не виден как модель.
    if ! mv -f "${dest}.tmp" "${dest}"; then
        warn "${where}: не удалось заменить ${dest} через mv -f"
        _unlock
        failed=$((failed + 1))
        continue
    fi
    mtime="$(_file_mtime "${dest}")"
    if [[ ${#SHA_CMD[@]} -gt 0 ]]; then
        final_sha="${sha}"
        if [[ -z "${final_sha}" ]]; then
            final_sha="$(_sha256 "${dest}" || true)"
        fi
        # Пустой хеш (сбой инструмента) не пишем в состояние: иначе следующий
        # boot считал бы файл «зафиксированным» без проверки.
        if [[ -n "${final_sha}" ]]; then
            _state_set "${key}" "${final_sha}" "${size}" "${mtime}"
        fi
    fi
    if [[ -n "${quarantine}" ]]; then
        rm -f "${quarantine}"
    fi
    _unlock
    downloaded=$((downloaded + 1))
    log "[${done_n}/${total}] готово ${key} (${size} байт)"
done <"${MANIFEST}"

elapsed=$(( $(date +%s) - started_at ))

if [[ "${failed}" -gt 0 ]]; then
    warn "не скачано/не проверено файлов: ${failed} из ${total}; ok: fast ${skipped_fast}, sha ${skipped_sha}, скачано ${downloaded}; ${elapsed}s; job'ы на отсутствующих моделях упадут с видимой ошибкой"
    exit 1
fi

log "готово: модели на ${MODELS_DIR} (fast ${skipped_fast}, sha ${skipped_sha}, скачано ${downloaded}, без проверки ${skipped_nosha}, VERIFY=off ${skipped_off}) за ${elapsed}s; свободно: $(df -h "${VOL_ROOT}" 2>/dev/null | awk 'NR==2{print $4}')"
