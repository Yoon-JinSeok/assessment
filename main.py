"""스트림릿 기반 성취평가 등급컷 예측 프로그램."""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import openpyxl  # type: ignore # noqa: F401

    OPENPYXL_AVAILABLE = True
except ImportError:  # pragma: no cover - 환경 의존
    OPENPYXL_AVAILABLE = False

GRADE_ORDER = ["A", "B", "C", "D", "E"]
GRADE_CUT_KEYS = ["A", "B", "C", "D"]
DEFAULT_CUTS = {"A": 90.0, "B": 80.0, "C": 70.0, "D": 60.0}
DEFAULT_TARGET = {"A": 20.0, "B": 30.0, "C": 30.0, "D": 15.0, "E": 5.0}


st.set_page_config(
    page_title="성취평가 등급컷 예측 프로그램",
    page_icon="📊",
    layout="wide",
)


def to_float(value) -> float | None:
    """엑셀 셀 값을 부동소수점으로 변환."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def normalize_identifier(value) -> str:
    """학생 식별에 사용할 문자열을 정규화."""

    if value is None:
        return ""
    if isinstance(value, (int, float)) and not np.isnan(value):
        if float(value).is_integer():
            return f"{int(value):03d}"
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return str(value).strip()


def make_student_key(class_label, student_no) -> str:
    class_id = normalize_identifier(class_label)
    student_id = normalize_identifier(student_no)
    if class_id:
        return f"{class_id}-{student_id}"
    return student_id


@st.cache_data(show_spinner=False)
def parse_gradebook(
    file_bytes: bytes,
    source_name: str,
    *,
    data_start_row_idx: int = 5,
    class_row_idx: int = 4,
) -> pd.DataFrame:
    """지정된 양식의 엑셀 파일을 DataFrame으로 변환."""

    if not OPENPYXL_AVAILABLE:
        raise ImportError(
            "openpyxl 라이브러리가 필요합니다. requirements.txt에 openpyxl을 추가하고 설치해 주세요."
        )

    df = pd.read_excel(io.BytesIO(file_bytes), header=None, engine="openpyxl")
    if df.empty:
        return pd.DataFrame(columns=["student_key", "score", "source"])

    start_row = data_start_row_idx
    end_row = df.shape[0]
    for idx in range(start_row, df.shape[0]):
        marker = df.iloc[idx, 0]
        if isinstance(marker, str) and "응시생수" in marker:
            end_row = idx
            break

    student_numbers = df.iloc[start_row:end_row, 0].tolist()
    class_labels = df.iloc[class_row_idx, 1:].tolist() if df.shape[0] > class_row_idx else []

    records: List[Dict[str, object]] = []
    for col_offset, class_label in enumerate(class_labels, start=1):
        if class_label is None or str(class_label).strip() == "":
            continue
        column_scores = df.iloc[start_row:end_row, col_offset].tolist()
        for row_idx, raw_score in enumerate(column_scores):
            student_no = student_numbers[row_idx] if row_idx < len(student_numbers) else None
            numeric_score = to_float(raw_score)
            if student_no is None or numeric_score is None:
                continue
            records.append(
                {
                    "student_key": make_student_key(class_label, student_no),
                    "class_label": class_label,
                    "student_no": student_no,
                    "score": numeric_score,
                    "source": source_name,
                }
            )

    return pd.DataFrame(records)


def build_score_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    return df.groupby("student_key")["score"].mean()


def apply_weights(
    student_scores: pd.DataFrame,
    maxima: Dict[str, float],
    weights: Dict[str, float],
) -> pd.DataFrame:
    def component(score: float, max_score: float, weight: float) -> float:
        if max_score <= 0 or score is None or np.isnan(score):
            return 0.0
        return (score / max_score) * weight

    student_scores["midterm_comp"] = student_scores["midterm"].apply(
        lambda val: component(val, maxima["midterm"], weights["midterm"])
    )
    student_scores["performance_comp"] = student_scores["performance"].apply(
        lambda val: component(val, maxima["performance"], weights["performance"])
    )
    student_scores["final_exam_comp"] = student_scores["final_exam"].apply(
        lambda val: component(val, maxima["final_exam"], weights["final_exam"])
    )
    student_scores["total"] = (
        student_scores["midterm_comp"]
        + student_scores["performance_comp"]
        + student_scores["final_exam_comp"]
    )
    return student_scores


def assign_grade(score: float, cuts: Dict[str, float]) -> str:
    if score is None or np.isnan(score):
        return "E"
    for grade in GRADE_CUT_KEYS:
        if score >= cuts.get(grade, 0.0):
            return grade
    return "E"


def render_grade_cut_inputs(section: str, max_score: float, defaults: Dict[str, float]) -> Dict[str, float]:
    cols = st.columns(len(GRADE_CUT_KEYS))
    cuts: Dict[str, float] = {}
    ceiling = max_score
    for idx, grade in enumerate(GRADE_CUT_KEYS):
        default_value = min(defaults.get(grade, ceiling), ceiling)
        cuts[grade] = cols[idx].number_input(
            f"{section} {grade}컷",
            min_value=0.0,
            max_value=float(ceiling),
            value=float(default_value),
            step=0.5,
            key=f"{section}_{grade}_cut",
        )
        ceiling = cuts[grade]
    return cuts


def render_grade_cut_sliders(
    label: str, max_score: float, defaults: Dict[str, float]
) -> Dict[str, float]:
    cols = st.columns(len(GRADE_CUT_KEYS))
    cuts: Dict[str, float] = {}
    ceiling = max_score
    for idx, grade in enumerate(GRADE_CUT_KEYS):
        default_value = min(defaults.get(grade, ceiling), ceiling)
        cuts[grade] = cols[idx].slider(
            f"{label} {grade}컷",
            min_value=0.0,
            max_value=float(ceiling),
            value=float(default_value),
            step=0.5,
            key=f"{label}_{grade}_slider",
        )
        ceiling = cuts[grade]
    return cuts


def summarize_distribution(
    totals: pd.Series, final_cuts: Dict[str, float], target_ratio: Dict[str, float]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    grades = totals.apply(lambda score: assign_grade(score, final_cuts))
    counts = grades.value_counts().reindex(GRADE_ORDER, fill_value=0)
    population = counts.sum()
    percentages = (counts / population * 100).fillna(0.0)
    summary = pd.DataFrame(
        {
            "등급": GRADE_ORDER,
            "학생 수": counts.values,
            "비율(%)": percentages.round(2).values,
            "목표 비율(%)": [target_ratio.get(grade, 0.0) for grade in GRADE_ORDER],
        }
    )
    summary["차이(%)"] = (summary["비율(%)"] - summary["목표 비율(%)"]).round(2)
    return summary, counts.to_dict()


def collect_target_ratio() -> Dict[str, float]:
    st.sidebar.subheader("목표 등급 비율(%)")
    ratio_inputs: Dict[str, float] = {}
    for grade in GRADE_ORDER:
        ratio_inputs[grade] = st.sidebar.number_input(
            f"{grade}",
            min_value=0.0,
            max_value=100.0,
            value=float(DEFAULT_TARGET.get(grade, 0.0)),
            step=1.0,
            key=f"target_{grade}",
        )
    total_ratio = sum(ratio_inputs.values())
    if total_ratio != 100.0:
        st.sidebar.warning(f"현재 입력된 목표 비율 합계는 {total_ratio:.1f}% 입니다. 100%가 되도록 조정하세요.")
    return ratio_inputs


def main() -> None:
    st.title("성취평가 등급컷 예측 프로그램")
    st.caption(
        "기말고사 성적 분포를 가정(중간고사와 동일)하여 원하는 학기말 성취평가 비율을 시뮬레이션합니다."
    )

    with st.sidebar:
        st.header("평가 기본 설정")
        num_performances = st.number_input(
            "수행평가 횟수",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )

        st.subheader("만점 및 반영 비율")
        midterm_max = st.number_input("중간고사 만점", min_value=1.0, value=100.0, step=1.0)
        final_exam_max = st.number_input("기말고사 만점", min_value=1.0, value=100.0, step=1.0)
        performance_max = st.number_input("수행평가 만점", min_value=1.0, value=100.0, step=1.0)

        midterm_weight = st.number_input("중간고사 반영비율(%)", min_value=0.0, max_value=100.0, value=30.0)
        final_exam_weight = st.number_input("기말고사 반영비율(%)", min_value=0.0, max_value=100.0, value=40.0)
        performance_weight = st.number_input("수행평가 반영비율(%)", min_value=0.0, max_value=100.0, value=30.0)

        weight_total = midterm_weight + final_exam_weight + performance_weight
        if abs(weight_total - 100.0) > 1e-6:
            st.warning(f"반영비율 합계가 {weight_total:.1f}% 입니다. 100%가 되도록 조정하세요.")

        st.subheader("중간고사 등급컷")
        midterm_cuts = render_grade_cut_inputs("중간", midterm_max, DEFAULT_CUTS)

        st.subheader("수행평가 등급컷")
        performance_cuts = render_grade_cut_inputs("수행", performance_max, DEFAULT_CUTS)

    target_ratio = collect_target_ratio()

    st.header("성적 파일 업로드")
    midterm_file = st.file_uploader("중간고사 성적 파일 (.xlsx)", type=["xlsx"], key="midterm_uploader")

    midterm_df = None
    if midterm_file is not None:
        midterm_bytes = midterm_file.read()
        try:
            midterm_df = parse_gradebook(midterm_bytes, "중간고사")
        except ImportError as exc:
            st.error(str(exc))
            st.stop()
        st.success(f"중간고사 데이터 {midterm_df['student_key'].nunique()}명 로드 완료")
    else:
        st.info("중간고사 성적 파일을 업로드해 주세요.")

    performance_files = st.file_uploader(
        "수행평가 성적 파일들 (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        key="performance_uploader",
    )

    performance_df = None
    if performance_files:
        perf_frames = []
        for idx, file in enumerate(performance_files, start=1):
            perf_bytes = file.read()
            try:
                parsed = parse_gradebook(
                    perf_bytes,
                    f"수행평가 {idx}",
                    data_start_row_idx=6,  # 엑셀 7행부터 학생 점수
                    class_row_idx=5,  # 엑셀 6행 반 정보
                )
            except ImportError as exc:
                st.error(str(exc))
                st.stop()
            parsed["assessment"] = idx
            perf_frames.append(parsed)
            st.info(f"수행평가 {idx}: {parsed['student_key'].nunique()}명")
        performance_df = pd.concat(perf_frames, ignore_index=True)
        if len(performance_files) != num_performances:
            st.warning(
                f"지정한 수행평가 횟수({num_performances})와 업로드된 파일 수({len(performance_files)})가 다릅니다."
            )
    else:
        st.info("수행평가 파일들을 모두 업로드해 주세요.")

    if midterm_df is None or midterm_df.empty:
        st.stop()

    midterm_series = build_score_series(midterm_df)
    performance_series = build_score_series(performance_df) if performance_df is not None else pd.Series(dtype="float64")

    all_students = sorted(set(midterm_series.index).union(performance_series.index))
    student_records = pd.DataFrame({"student_key": all_students})
    student_records["midterm"] = student_records["student_key"].map(midterm_series)
    student_records["performance"] = student_records["student_key"].map(performance_series)
    student_records["final_exam"] = student_records["midterm"]  # 기말고사 = 중간고사 가정

    student_records = apply_weights(
        student_records,
        {
            "midterm": midterm_max,
            "performance": performance_max,
            "final_exam": final_exam_max,
        },
        {
            "midterm": midterm_weight,
            "performance": performance_weight,
            "final_exam": final_exam_weight,
        },
    )

    st.header("기말고사 등급컷 시뮬레이션")
    st.write("기본값은 중간고사 등급컷을 따릅니다. 각 슬라이더를 드래그하면 즉시 재계산됩니다.")
    default_final_cuts = midterm_cuts.copy()
    final_exam_cuts = render_grade_cut_sliders("기말", final_exam_max, default_final_cuts)

    if final_exam_max <= 0:
        st.error("기말고사 만점은 0보다 커야 합니다.")
        st.stop()

    total_weight_max = midterm_weight + performance_weight + final_exam_weight
    exam_cut_percentages = {
        grade: (final_exam_cuts.get(grade, 0.0) / final_exam_max) for grade in GRADE_CUT_KEYS
    }
    final_total_cuts = {
        grade: exam_cut_percentages[grade] * total_weight_max for grade in GRADE_CUT_KEYS
    }

    summary_df, counts = summarize_distribution(student_records["total"], final_total_cuts, target_ratio)

    st.subheader("학기말 성취평가 분포")
    cols = st.columns(3)
    cols[0].metric("전체 학생 수", f"{int(summary_df['학생 수'].sum())}명")
    valid_total = student_records["total"].notna().sum()
    cols[1].metric("총점 데이터 확보율", f"{valid_total / len(student_records) * 100:.1f}%")
    cols[2].metric("기말 등급컷 기준", f"{final_exam_cuts['A']:.1f}/{final_exam_max:.0f}")

    st.dataframe(summary_df, use_container_width=True)

    chart_source = summary_df.set_index("등급")["비율(%)"]
    comparison = summary_df.set_index("등급")[["비율(%)", "목표 비율(%)"]]
    st.bar_chart(comparison)

    st.subheader("학생별 가중 총점 미리보기")
    st.dataframe(
        student_records.sort_values("total", ascending=False)
        .head(15)
        .assign(등급=lambda df_: df_["total"].apply(lambda v: assign_grade(v, final_total_cuts)))
        [["student_key", "midterm", "performance", "final_exam", "total", "등급"]]
        .rename(
            columns={
                "student_key": "학생ID",
                "midterm": "중간고사",
                "performance": "수행평가",
                "final_exam": "기말고사",
                "total": "학기말총점",
            }
        ),
        use_container_width=True,
    )

    st.caption(
        "※ 기말고사 성적은 중간고사와 동일하다고 가정했으며, 기말 등급컷 슬라이더 비율을 학기말 총점에도 동일하게 적용했습니다."
    )


if __name__ == "__main__":
    main()
