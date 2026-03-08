#!/usr/bin/env bash
set -euo pipefail

RUN_CASE_TEMPLATE="run_case.py"
RUN_CASE_TMP="run_case_tmp.py"

run_mode() {
    local MODE="$1"
    local CASES_FILE="$2"
    local OUTPUT_CSV="$3"
    local OUTPUT_PNG="$4"

    if [ ! -f "$CASES_FILE" ]; then
        echo "Erreur : fichier introuvable -> $CASES_FILE"
        exit 1
    fi

    rm -f "$OUTPUT_CSV"

    echo ""
    echo "=============================="
    echo " Lancement du mode : $MODE"
    echo " Fichier de cas     : $CASES_FILE"
    echo " Sortie CSV         : $OUTPUT_CSV"
    echo " Sortie PNG         : $OUTPUT_PNG"
    echo "=============================="
    echo ""

    while IFS=' ' read -r N_PROFILE DT || [ -n "${N_PROFILE:-}" ]; do
        if [[ -z "${N_PROFILE:-}" ]] || [[ "${N_PROFILE}" =~ ^# ]]; then
            continue
        fi

        echo "=== MODE=${MODE} | n_profile=${N_PROFILE} | dt=${DT} ==="

        cp "$RUN_CASE_TEMPLATE" "$RUN_CASE_TMP"

        sed -i "s/MODE_PLACEHOLDER/${MODE}/g" "$RUN_CASE_TMP"
        sed -i "s/N_PROFILE_PLACEHOLDER/${N_PROFILE}/g" "$RUN_CASE_TMP"
        sed -i "s/DT_PLACEHOLDER/${DT}/g" "$RUN_CASE_TMP"
        sed -i "s|OUTPUT_CSV_PLACEHOLDER|${OUTPUT_CSV}|g" "$RUN_CASE_TMP"

        python.exe "$RUN_CASE_TMP" < /dev/null

        rm -f "$RUN_CASE_TMP"
    done < "$CASES_FILE"

    python.exe analyse_de_convergence.py \
        --input "$OUTPUT_CSV" \
        --mode "$MODE" \
        --output "$OUTPUT_PNG"
}

run_mode "space" "space_cases.txt" "results_space.csv" "convergence_space.png"
run_mode "time" "time_cases.txt" "results_time.csv" "convergence_time.png"

echo ""
echo "Toutes les analyses sont terminées."
echo "Figures générées :"
echo "  - convergence_space.png"
echo "  - convergence_time.png"
python.exe show_convergence_plots.py