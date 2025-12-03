"""스트림릿 기반 성취평가 등급컷 예측 프로그램."""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import altair as alt
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
PERFORMANCE_DEFAULT_CUTS = {"A": 36.0, "B": 32.0, "C": 28.0, "D": 24.0, "E": 21.0}
DEFAULT_TARGET = {"A": 32.0, "B": 40.0, "C": 20.0, "D": 5.0, "E": 3.0}


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


def collect_student_meta(*grade_frames: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """각 평가 데이터프레임에서 학생 별 반/번호 정보를 모은다."""

    meta: Dict[str, Dict[str, object]] = {}
    for frame in grade_frames:
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            key = row.get("student_key")
            if not key:
                continue
            entry = meta.setdefault(key, {"class_label": None, "student_no": None})
            class_val = row.get("class_label")
            if entry["class_label"] in (None, "") and class_val not in (None, ""):
                entry["class_label"] = class_val
            number_val = row.get("student_no")
            if entry["student_no"] in (None, "") and number_val not in (None, ""):
                entry["student_no"] = number_val
    return meta


def format_student_display(class_label, student_no, fallback: str = "") -> str:
    """반-번호 형식 문자열 생성."""

    class_part = normalize_identifier(class_label)
    student_part = normalize_identifier(student_no)
    if class_part and student_part:
        return f"{class_part}-{student_part}"
    if class_part:
        return class_part
    if student_part:
        return student_part
    return fallback or "-"


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


def build_score_series(df: pd.DataFrame, reducer: str = "mean") -> pd.Series:
    """평가 데이터프레임을 학생별 점수 시리즈로 변환."""

    if df is None or df.empty:
        return pd.Series(dtype="float64")

    grouped = df.groupby("student_key")["score"]
    if reducer == "sum":
        return grouped.sum()
    if reducer == "min":
        return grouped.min()
    if reducer == "max":
        return grouped.max()
    return grouped.mean()


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


def render_grade_cut_inputs(
    section: str,
    max_score: float,
    defaults: Dict[str, float],
    *,
    integer: bool = False,
) -> Dict[str, float]:
    cols = st.columns(len(GRADE_CUT_KEYS))
    cuts: Dict[str, float] = {}
    ceiling = max_score
    for idx, grade in enumerate(GRADE_CUT_KEYS):
        default_value = min(defaults.get(grade, ceiling), ceiling)
        if integer:
            default_value = int(round(default_value))
            cuts[grade] = cols[idx].number_input(
                f"{section} {grade}컷",
                min_value=0,
                max_value=int(ceiling),
                value=default_value,
                step=1,
                key=f"{section}_{grade}_cut",
            )
        else:
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
            min_value=0,
            max_value=100,
            value=int(DEFAULT_TARGET.get(grade, 0.0)),
            step=1,
            key=f"target_{grade}",
        )
    total_ratio = sum(ratio_inputs.values())
    if total_ratio != 100.0:
        st.sidebar.warning(f"현재 입력된 목표 비율 합계는 {total_ratio:.1f}% 입니다. 100%가 되도록 조정하세요.")
    return ratio_inputs


def main() -> None:
    st.title("성취평가 등급컷 예측 프로그램")
    st.caption(
        "기말고사 성적 분포를 가정(중간고사와 동일)하여 원하는 학기말 성취평가 비율을 시뮬레이션합니다. / created by 윤진석"
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
        midterm_max = st.number_input("중간고사 만점", min_value=1, value=100, step=1)
        final_exam_max = st.number_input("기말고사 만점", min_value=1, value=100, step=1)
        performance_max = st.number_input("수행평가 만점", min_value=1, value=40, step=1)

        midterm_weight = st.number_input("중간고사 반영비율(%)", min_value=0, max_value=100, value=30, step=1)
        final_exam_weight = st.number_input("기말고사 반영비율(%)", min_value=0, max_value=100, value=30, step=1)
        performance_weight = st.number_input("수행평가 반영비율(%)", min_value=0, max_value=100, value=40, step=1)

        weight_total = midterm_weight + final_exam_weight + performance_weight
        if abs(weight_total - 100.0) > 1e-6:
            st.warning(f"반영비율 합계가 {weight_total:.1f}% 입니다. 100%가 되도록 조정하세요.")

        st.subheader("중간고사 등급컷")
        midterm_cuts = render_grade_cut_inputs("중간", midterm_max, DEFAULT_CUTS, integer=True)

        st.subheader("수행평가 등급컷")
        performance_cuts = render_grade_cut_inputs(
            "수행", performance_max, PERFORMANCE_DEFAULT_CUTS, integer=True
        )

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

    midterm_series = build_score_series(midterm_df, reducer="mean")
    performance_series = build_score_series(performance_df, reducer="sum")

    all_students = sorted(set(midterm_series.index).union(performance_series.index))
    student_records = pd.DataFrame({"student_key": all_students})
    student_records["midterm"] = student_records["student_key"].map(midterm_series)
    student_records["performance"] = student_records["student_key"].map(performance_series)
    student_records["final_exam"] = student_records["midterm"]  # 기말고사 = 중간고사 가정

    student_meta = collect_student_meta(midterm_df, performance_df)
    class_map = {key: info.get("class_label") for key, info in student_meta.items()}
    number_map = {key: info.get("student_no") for key, info in student_meta.items()}
    student_records["class_label"] = student_records["student_key"].map(class_map)
    student_records["student_no"] = student_records["student_key"].map(number_map)

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

    student_records["student_display"] = [
        format_student_display(cls, num, fallback=key)
        for cls, num, key in zip(
            student_records["class_label"],
            student_records["student_no"],
            student_records["student_key"],
        )
    ]

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

    chart_df = summary_df.melt(
        id_vars="등급",
        value_vars=["비율(%)", "목표 비율(%)"],
        var_name="지표",
        value_name="비율값",
    )
    bar_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("등급:N", title="등급"),
            y=alt.Y("비율값:Q", title="비율(%)"),
            color=alt.Color("지표:N", title="구분"),
            xOffset="지표:N",
        )
        .properties(height=320)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    st.subheader("학생별 가중 총점")
    student_table = (
        student_records.copy()
        .assign(등급=lambda df_: df_["total"].apply(lambda v: assign_grade(v, final_total_cuts)))
        .rename(
            columns={
                "student_display": "학생ID",
                "midterm": "중간고사",
                "performance": "수행평가",
                "final_exam": "기말고사",
                "total": "학기말총점",
            }
        )
        .sort_values("학생ID")
    )
    st.dataframe(
        student_table[["학생ID", "중간고사", "수행평가", "기말고사", "학기말총점", "등급"]],
        use_container_width=True,
    )

    st.caption(
        "※ 기말고사 성적은 중간고사와 동일하다고 가정했으며, 기말 등급컷 슬라이더 비율을 학기말 총점에도 동일하게 적용했습니다."
    )


if __name__ == "__main__":
    main()
