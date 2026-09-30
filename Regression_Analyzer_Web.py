from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

APP_TITLE = "선형·비선형 회귀 분석 프로그램"
NONE = "(None)"

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")


def _safe_name(value: object, index: int) -> str:
    if pd.isna(value) or not str(value).strip() or str(value).lower().startswith("unnamed"):
        letters = ""
        number = index + 1
        while number:
            number, remainder = divmod(number - 1, 26)
            letters = chr(65 + remainder) + letters
        return f"Column_{letters}"
    return str(value).strip()


def make_unique(values: list[object]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for index, value in enumerate(values):
        base = _safe_name(value, index)
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return result


def _is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def guess_header_row(raw: pd.DataFrame, max_rows: int = 20) -> int:
    best_row, best_score = 0, -1.0
    for row_i in range(min(max_rows, len(raw))):
        row = raw.iloc[row_i]
        present = row.dropna()
        if present.empty:
            continue
        text_count = sum(not _is_number(value) for value in present)
        unique_count = present.astype(str).nunique()
        score = len(present) + 2.0 * text_count + 0.2 * unique_count
        if score > best_score:
            best_row, best_score = row_i, score
    return best_row


def r2_score(y: np.ndarray, predicted: np.ndarray) -> float:
    sse = float(np.sum((y - predicted) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return float("nan") if sst <= np.finfo(float).eps else 1.0 - sse / sst


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def exponential(x, a, b, c):
    with np.errstate(over="ignore", invalid="ignore"):
        return a * np.exp(np.clip(b * x, -700, 700)) + c


def logarithmic(x, a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a * np.log(x) + b


def power(x, a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return a * np.power(x, b)


def logistic4(x, lower, upper, midpoint, slope):
    with np.errstate(over="ignore", invalid="ignore"):
        return lower + (upper - lower) / (
            1.0 + np.exp(np.clip(-slope * (x - midpoint), -700, 700))
        )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    function: Callable
    parameters: tuple[str, ...]
    minimum_points: int
    positive_x: bool = False


MODELS: dict[str, ModelSpec] = {
    "Linear": ModelSpec("Linear", linear, ("a", "b"), 2),
    "Quadratic": ModelSpec("Quadratic", quadratic, ("a", "b", "c"), 3),
    "Cubic": ModelSpec("Cubic", cubic, ("a", "b", "c", "d"), 4),
    "Exponential": ModelSpec("Exponential", exponential, ("a", "b", "c"), 3),
    "Logarithmic": ModelSpec("Logarithmic", logarithmic, ("a", "b"), 2, True),
    "Power": ModelSpec("Power", power, ("a", "b"), 2, True),
    "Logistic 4P": ModelSpec("Logistic 4P", logistic4, ("lower", "upper", "midpoint", "slope"), 4),
}


@dataclass
class FitResult:
    case: str
    model: str
    spec: ModelSpec
    params: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_hat: np.ndarray
    residuals: np.ndarray
    n: int
    k: int
    r2: float
    adjusted_r2: float
    rmse: float
    mae: float
    sse: float
    aic: float
    aicc: float
    bic: float
    equation: str
    outlier_mask: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.spec.function(values, *self.params), dtype=float)


def initial_guess(name: str, x: np.ndarray, y: np.ndarray):
    span_x = max(float(np.ptp(x)), 1e-9)
    span_y = max(float(np.ptp(y)), 1e-9)
    if name == "Exponential":
        guess = np.array([span_y, 1.0 / span_x, float(np.min(y))])
        bounds = (
            np.array([-np.inf, -100.0 / span_x, -np.inf]),
            np.array([np.inf, 100.0 / span_x, np.inf]),
        )
    elif name == "Logistic 4P":
        corr = np.corrcoef(x, y)[0, 1]
        direction = 1.0 if np.isfinite(corr) and corr >= 0 else -1.0
        guess = np.array([float(np.min(y)), float(np.max(y)), float(np.median(x)), direction / span_x])
        bounds = (
            np.array([-np.inf, -np.inf, float(np.min(x)) - span_x, -100.0 / span_x]),
            np.array([np.inf, np.inf, float(np.max(x)) + span_x, 100.0 / span_x]),
        )
    elif name == "Power":
        lx, ly = np.log(x), np.log(np.clip(np.abs(y), 1e-12, None))
        slope, intercept = np.polyfit(lx, ly, 1)
        guess = np.array([float(np.exp(intercept)) * (1 if np.mean(y) >= 0 else -1), float(slope)])
        bounds = (np.array([-np.inf, -100.0]), np.array([np.inf, 100.0]))
    else:
        guess = np.ones(len(MODELS[name].parameters))
        bounds = (np.full(len(guess), -np.inf), np.full(len(guess), np.inf))
    return guess, bounds


def equation_text(model: str, p: np.ndarray) -> str:
    f = lambda v: f"{v:.6g}"
    if model == "Linear":
        return f"y = {f(p[0])}x + {f(p[1])}"
    if model == "Quadratic":
        return f"y = {f(p[0])}x² + {f(p[1])}x + {f(p[2])}"
    if model == "Cubic":
        return f"y = {f(p[0])}x³ + {f(p[1])}x² + {f(p[2])}x + {f(p[3])}"
    if model == "Exponential":
        return f"y = {f(p[0])} exp({f(p[1])}x) + {f(p[2])}"
    if model == "Logarithmic":
        return f"y = {f(p[0])} ln(x) + {f(p[1])}"
    if model == "Power":
        return f"y = {f(p[0])} x^{f(p[1])}"
    return f"y = {f(p[0])} + ({f(p[1])} - {f(p[0])}) / (1 + exp(-{f(p[3])}(x - {f(p[2])})))"


def fit_model(case: str, model_name: str, x: np.ndarray, y: np.ndarray) -> FitResult:
    spec = MODELS[model_name]
    mask = np.isfinite(x) & np.isfinite(y)
    if spec.positive_x:
        mask &= x > 0
    x, y = x[mask], y[mask]
    n, k = len(x), len(spec.parameters)

    if n <= k or n < spec.minimum_points:
        raise ValueError(f"{model_name}: at least {k + 1} valid points are required")
    if np.unique(x).size < spec.minimum_points:
        raise ValueError(f"{model_name}: not enough unique X values")

    if model_name in ("Linear", "Quadratic", "Cubic"):
        degree = {"Linear": 1, "Quadratic": 2, "Cubic": 3}[model_name]
        params, covariance = np.polyfit(x, y, degree, cov=True)
    else:
        guess, bounds = initial_guess(model_name, x, y)
        params, covariance = curve_fit(
            spec.function, x, y, p0=guess, bounds=bounds,
            maxfev=100000, method="trf"
        )

    y_hat = np.asarray(spec.function(x, *params), dtype=float)
    if not np.all(np.isfinite(y_hat)):
        raise ValueError(f"{model_name}: fitted values are not finite")

    residuals = y - y_hat
    sse = float(np.sum(residuals**2))
    mse = sse / n
    r2 = r2_score(y, y_hat)
    adjusted = float("nan") if n <= k else 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(residuals)))
    safe_mse = max(mse, np.finfo(float).tiny)
    aic = n * math.log(safe_mse) + 2 * k
    aicc = float("inf") if n <= k + 1 else aic + (2 * k * (k + 1)) / (n - k - 1)
    bic = n * math.log(safe_mse) + k * math.log(n)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    residual_sd = math.sqrt(sse / max(n - k, 1))
    standardized = residuals / max(residual_sd, np.finfo(float).eps)
    outliers = np.abs(standardized) >= 2.0

    return FitResult(
        case=case, model=model_name, spec=spec, params=params,
        covariance=covariance, standard_errors=standard_errors,
        x=x, y=y, y_hat=y_hat, residuals=residuals,
        n=n, k=k, r2=r2, adjusted_r2=adjusted,
        rmse=rmse, mae=mae, sse=sse, aic=aic, aicc=aicc, bic=bic,
        equation=equation_text(model_name, params), outlier_mask=outliers
    )


def confidence_band(result: FitResult, x_grid: np.ndarray, level: float = 0.95):
    predicted = result.predict(x_grid)
    p = result.params
    jacobian = np.empty((len(x_grid), len(p)), dtype=float)
    for index in range(len(p)):
        step = math.sqrt(np.finfo(float).eps) * max(abs(float(p[index])), 1.0)
        upper_p, lower_p = p.copy(), p.copy()
        upper_p[index] += step
        lower_p[index] -= step
        jacobian[:, index] = (
            result.spec.function(x_grid, *upper_p)
            - result.spec.function(x_grid, *lower_p)
        ) / (2.0 * step)
    variances = np.einsum("ij,jk,ik->i", jacobian, result.covariance, jacobian)
    se = np.sqrt(np.maximum(variances, 0.0))
    critical = student_t.ppf((1.0 + level) / 2.0, max(result.n - result.k, 1))
    return predicted - critical * se, predicted + critical * se


def read_uploaded_raw(uploaded_file, sheet_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]
    data = uploaded_file.getvalue()
    if suffix == "csv":
        try:
            raw = pd.read_csv(io.BytesIO(data), header=None, encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = pd.read_csv(io.BytesIO(data), header=None, encoding="cp949")
        return raw.dropna(axis=1, how="all"), []
    excel = pd.ExcelFile(io.BytesIO(data))
    sheets = excel.sheet_names
    selected = sheet_name or sheets[0]
    raw = pd.read_excel(io.BytesIO(data), sheet_name=selected, header=None)
    return raw.dropna(axis=1, how="all"), sheets


def apply_header(raw: pd.DataFrame, header_row_1based: int) -> pd.DataFrame:
    row = header_row_1based - 1
    if row < 0 or row >= len(raw):
        raise ValueError(f"Header row must be between 1 and {len(raw)}")
    columns = make_unique(raw.iloc[row].tolist())
    data = raw.iloc[row + 1:].copy()
    data.columns = columns
    return data.dropna(how="all").reset_index(drop=True)


def analysis_groups(data: pd.DataFrame, group_col: str):
    if group_col == NONE:
        return [("All", data)]
    return [(str(name), frame.copy()) for name, frame in data.groupby(group_col, dropna=False, sort=False)]


def best_results(results: list[FitResult]) -> dict[str, FitResult]:
    best: dict[str, FitResult] = {}
    for result in results:
        score = result.aicc if np.isfinite(result.aicc) else result.aic
        current = best.get(result.case)
        current_score = (
            current.aicc if current and np.isfinite(current.aicc)
            else current.aic if current else float("inf")
        )
        if score < current_score:
            best[result.case] = result
    return best


def build_summary_df(results: list[FitResult]) -> pd.DataFrame:
    best = best_results(results)
    rows = []
    for r in results:
        rows.append({
            "Best": "★" if best.get(r.case) is r else "",
            "Case": r.case,
            "Model": r.model,
            "N": r.n,
            "R²": r.r2,
            "Adjusted R²": r.adjusted_r2,
            "RMSE": r.rmse,
            "MAE": r.mae,
            "SSE": r.sse,
            "AIC": r.aic,
            "AICc": r.aicc,
            "BIC": r.bic,
            "Equation": r.equation,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Case", "Best", "AICc"], ascending=[True, False, True]).reset_index(drop=True)


def build_parameter_df(results: list[FitResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for name, value, se in zip(r.spec.parameters, r.params, r.standard_errors):
            rows.append({
                "Case": r.case, "Model": r.model, "Parameter": name,
                "Estimate": value, "Standard Error": se
            })
    return pd.DataFrame(rows)


def build_predictions_df(results: list[FitResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        residual_sd = math.sqrt(r.sse / max(r.n - r.k, 1))
        standardized = r.residuals / max(residual_sd, np.finfo(float).eps)
        for i, (xv, yv, pred, resid, stdres, outlier) in enumerate(
            zip(r.x, r.y, r.y_hat, r.residuals, standardized, r.outlier_mask), start=1
        ):
            rows.append({
                "Case": r.case, "Model": r.model, "Index": i,
                "X": xv, "Observed Y": yv, "Predicted Y": pred,
                "Residual": resid, "Standardized Residual": stdres,
                "Outlier": bool(outlier),
            })
    return pd.DataFrame(rows)


def build_best_residual_df(results: list[FitResult]) -> pd.DataFrame:
    best = best_results(results)
    pred = build_predictions_df(results)
    if pred.empty:
        return pred
    mask = pred.apply(lambda row: best.get(row["Case"]) is not None and best[row["Case"]].model == row["Model"], axis=1)
    return pred[mask].reset_index(drop=True)


def export_excel_bytes(results: list[FitResult], failures: list[dict], data: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        build_summary_df(results).to_excel(writer, sheet_name="Model Summary", index=False)
        build_parameter_df(results).to_excel(writer, sheet_name="Parameters", index=False)
        build_predictions_df(results).to_excel(writer, sheet_name="All Predictions", index=False)
        build_best_residual_df(results).to_excel(writer, sheet_name="Best Residuals", index=False)
        build_algorithm_df().to_excel(writer, sheet_name="Algorithm", index=False)
        if failures:
            pd.DataFrame(failures).to_excel(writer, sheet_name="Failed Models", index=False)
        data.to_excel(writer, sheet_name="Input Data", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cells in sheet.columns:
                width = min(max(len(str(c.value)) if c.value is not None else 0 for c in cells) + 2, 55)
                sheet.column_dimensions[cells[0].column_letter].width = width
    return output.getvalue()


def regression_figure(
    results: list[FitResult], data: pd.DataFrame,
    x_col: str, y_col: str, error_col: str,
    group_col: str, show_ci: bool, show_outliers: bool
):
    fig = go.Figure()
    best = best_results(results)
    groups = dict(analysis_groups(data, group_col))

    for case in dict.fromkeys(r.case for r in results):
        case_results = [r for r in results if r.case == case]
        base = case_results[0]
        frame = groups.get(case)

        if frame is not None and error_col != NONE:
            xx = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(float)
            yy = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(float)
            ee = pd.to_numeric(frame[error_col], errors="coerce").to_numpy(float)
            valid = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ee)
            fig.add_trace(go.Scatter(
                x=xx[valid], y=yy[valid], mode="markers", name=f"{case} data",
                error_y=dict(type="data", array=np.abs(ee[valid]), visible=True),
                hovertemplate="X=%{x}<br>Y=%{y}<extra></extra>"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=base.x, y=base.y, mode="markers", name=f"{case} data",
                hovertemplate="X=%{x}<br>Y=%{y}<extra></extra>"
            ))

        lo, hi = float(np.min(base.x)), float(np.max(base.x))
        x_grid = np.linspace(lo, hi, 500)
        for result in case_results:
            y_grid = result.predict(x_grid)
            is_best = best.get(case) is result
            fig.add_trace(go.Scatter(
                x=x_grid, y=y_grid, mode="lines",
                name=f"{case} · {result.model}{' ★' if is_best else ''}",
                line=dict(width=4 if is_best else 2),
                hovertemplate=f"{result.model}<br>X=%{{x:.5g}}<br>Y=%{{y:.5g}}<extra></extra>"
            ))
            if is_best and show_ci:
                try:
                    lower, upper = confidence_band(result, x_grid)
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x_grid, x_grid[::-1]]),
                        y=np.concatenate([upper, lower[::-1]]),
                        fill="toself", mode="lines", line=dict(width=0),
                        opacity=0.15, name=f"{case} · 95% CI",
                        hoverinfo="skip"
                    ))
                except Exception:
                    pass

            if is_best and show_outliers and np.any(result.outlier_mask):
                idx = np.flatnonzero(result.outlier_mask)
                residual_sd = math.sqrt(result.sse / max(result.n - result.k, 1))
                std = result.residuals[idx] / max(residual_sd, np.finfo(float).eps)
                custom = np.column_stack([
                    result.y[idx], result.y_hat[idx], result.residuals[idx], std
                ])
                fig.add_trace(go.Scatter(
                    x=result.x[idx], y=result.y[idx], mode="markers",
                    name=f"{case} · outlier",
                    marker=dict(symbol="circle-open", size=14, line=dict(width=2)),
                    customdata=custom,
                    hovertemplate=(
                        "X=%{x:.6g}<br>Observed Y=%{customdata[0]:.6g}"
                        "<br>Predicted Y=%{customdata[1]:.6g}"
                        "<br>Residual=%{customdata[2]:+.6g}"
                        "<br>Std. residual=%{customdata[3]:+.3f}<extra></extra>"
                    )
                ))

    fig.update_layout(
        title="Regression Model Comparison",
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="closest",
        legend_title="Series",
        height=650,
        margin=dict(l=30, r=20, t=60, b=30),
    )
    return fig


def residual_figure(results: list[FitResult]):
    fig = go.Figure()
    for case, result in best_results(results).items():
        residual_sd = math.sqrt(result.sse / max(result.n - result.k, 1))
        std = result.residuals / max(residual_sd, np.finfo(float).eps)
        custom = np.column_stack([result.x, result.y, std, result.outlier_mask.astype(int)])
        fig.add_trace(go.Scatter(
            x=result.y_hat, y=result.residuals, mode="markers",
            name=f"{case} · {result.model}", customdata=custom,
            hovertemplate=(
                "X=%{customdata[0]:.6g}<br>Observed Y=%{customdata[1]:.6g}"
                "<br>Predicted Y=%{x:.6g}<br>Residual=%{y:+.6g}"
                "<br>Std. residual=%{customdata[2]:+.3f}<extra></extra>"
            )
        ))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Residual Plot · Best Model per Case",
        xaxis_title="Predicted Y",
        yaxis_title="Residual (Observed − Predicted)",
        height=560,
        margin=dict(l=30, r=20, t=60, b=30),
    )
    return fig


def run_models(data, x_col, y_col, group_col, selected_models):
    results: list[FitResult] = []
    failures: list[dict] = []
    for case, frame in analysis_groups(data, group_col):
        x_all = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
        y_all = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
        for name in selected_models:
            try:
                results.append(fit_model(case, name, x_all, y_all))
            except Exception as exc:
                failures.append({"Case": case, "Model": name, "Reason": str(exc)})
    return results, failures



MODEL_KR = {
    "Linear": "선형 회귀",
    "Quadratic": "2차 다항 회귀",
    "Cubic": "3차 다항 회귀",
    "Exponential": "지수 회귀",
    "Logarithmic": "로그 회귀",
    "Power": "거듭제곱 회귀",
    "Logistic 4P": "4-모수 로지스틱 회귀",
}


def build_algorithm_df() -> pd.DataFrame:
    """과제 설명용: 프로그램 내부에서 실제 사용하는 처리 알고리즘 요약."""
    return pd.DataFrame([
        {
            "단계": "1. 데이터 전처리",
            "알고리즘": "CSV/Excel 읽기 → Header 적용 → X/Y 숫자 변환 → 결측/비수치 제거",
            "구현": "Pandas",
        },
        {
            "단계": "2. 선형·다항 회귀",
            "알고리즘": "잔차제곱합(SSE)이 최소가 되도록 회귀계수 계산",
            "구현": "NumPy polyfit (최소제곱 적합)",
        },
        {
            "단계": "3. 비선형 회귀",
            "알고리즘": "초기 파라미터에서 시작해 잔차제곱합이 감소하도록 반복 최적화",
            "구현": "SciPy curve_fit + Trust Region Reflective(TRF)",
        },
        {
            "단계": "4. 모델 평가",
            "알고리즘": "예측값과 관측값의 잔차로 R², RMSE, MAE, SSE, AIC/AICc/BIC 계산",
            "구현": "NumPy / Python",
        },
        {
            "단계": "5. 최적 모델 선택",
            "알고리즘": "계산 가능한 모델 중 AICc 최소 모델 선택, AICc 불가 시 AIC 최소 모델 선택",
            "구현": "best_results()",
        },
        {
            "단계": "6. 이상치 후보",
            "알고리즘": "표준화 잔차의 절댓값이 2 이상인 점을 후보로 표시",
            "구현": "|standardized residual| ≥ 2",
        },
        {
            "단계": "7. 신뢰구간·예측",
            "알고리즘": "파라미터 공분산과 t 분포로 95% 신뢰구간 근사, 최적 회귀식으로 새 X의 Y 예측",
            "구현": "Jacobian + covariance + Student t",
        },
    ])


def render_algorithm_overview(selected_models: list[str] | None = None) -> None:
    """Streamlit 화면에 과제 제출용 알고리즘 설명을 표시."""
    st.markdown("### 전체 처리 알고리즘")
    st.markdown(
        "**데이터 입력 → 전처리 → 회귀모델 적합 → 예측값 계산 → 잔차·평가지표 계산 "
        "→ 최적 모델 선택 → 이상치/신뢰구간 확인 → 예측 및 결과 저장**"
    )
    st.dataframe(build_algorithm_df(), use_container_width=True, hide_index=True)

    st.markdown("#### 선형·다항 회귀 알고리즘")
    st.markdown(
        "선형, 2차, 3차 다항 회귀는 **최소제곱법(Least Squares)** 을 사용한다. "
        "관측값 `y`와 회귀식의 예측값 `ŷ` 차이인 잔차 `e = y - ŷ`의 제곱합 "
        "`SSE = Σ(y - ŷ)²`가 최소가 되도록 계수를 계산하며, 프로그램에서는 `numpy.polyfit`을 사용한다. "
        "2차·3차 모델은 그래프 모양은 곡선이지만 계수에 대해서는 선형인 다항 회귀이다."
    )

    st.markdown("#### 비선형 회귀 알고리즘")
    st.markdown(
        "지수, 로그, 거듭제곱, 4-모수 로지스틱 회귀는 `scipy.optimize.curve_fit`을 사용한다. "
        "초기 파라미터를 설정한 뒤 **Trust Region Reflective(TRF)** 방식의 비선형 최소제곱 최적화를 반복하여 "
        "SSE가 작아지는 파라미터를 찾는다. 로그·거듭제곱 모델은 `X > 0` 데이터만 사용한다."
    )

    st.markdown("#### 최적 모델 선택 알고리즘")
    st.markdown(
        "각 모델의 적합이 끝나면 R², 수정 R², RMSE, MAE, SSE, AIC, AICc, BIC를 계산한다. "
        "프로그램의 **최고 모델(Best model)** 표시는 AICc가 계산 가능한 경우 **AICc가 가장 낮은 모델**, "
        "그렇지 않은 경우 **AIC가 가장 낮은 모델**을 선택한다."
    )

    if selected_models:
        linear_like = [MODEL_KR[m] for m in selected_models if m in {"Linear", "Quadratic", "Cubic"}]
        nonlinear = [MODEL_KR[m] for m in selected_models if m not in {"Linear", "Quadratic", "Cubic"}]
        st.markdown("#### 현재 선택된 모델에 적용되는 알고리즘")
        if linear_like:
            st.write("- 최소제곱 다항식 적합: " + ", ".join(linear_like))
        if nonlinear:
            st.write("- 비선형 최소제곱 최적화(TRF): " + ", ".join(nonlinear))


# ---------- UI ----------
st.title("📈 선형·비선형 회귀 분석 프로그램")
st.caption("공개 또는 실험 데이터를 입력하여 선형회귀와 비선형회귀를 수행하고 계산 알고리즘과 결과를 함께 확인하는 웹 프로그램")

with st.expander("📌 사용 순서", expanded=False):
    st.markdown(
        "1. CSV 또는 Excel 파일 업로드  →  2. 열 제목 행 확인  →  "
        "3. X/Y 열 선택  →  4. 회귀모델 선택  →  5. 분석 실행  →  "
        "6. 그래프·잔차·지표·알고리즘 확인  →  7. 결과 다운로드"
    )

with st.expander("🧠 프로그램 알고리즘 설명", expanded=False):
    render_algorithm_overview()

uploaded = st.file_uploader("CSV / Excel 파일 업로드", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("분석할 CSV 또는 Excel 파일을 업로드하세요.")
    st.stop()

suffix = uploaded.name.lower().rsplit(".", 1)[-1]
sheet_name = None
sheets: list[str] = []
if suffix in ("xlsx", "xls"):
    _, sheets = read_uploaded_raw(uploaded)
    sheet_name = st.selectbox("Excel 시트", sheets)

raw, _ = read_uploaded_raw(uploaded, sheet_name)
if raw.empty:
    st.error("선택한 파일/시트에 데이터가 없습니다.")
    st.stop()

guessed = guess_header_row(raw) + 1
max_header = max(1, len(raw))
header_row = st.number_input(
    "열 제목 행 (Header row)",
    min_value=1, max_value=max_header, value=min(guessed, max_header), step=1
)

with st.expander("원본 데이터 미리보기", expanded=True):
    preview = raw.head(100).copy()
    preview.index = np.arange(1, len(preview) + 1)
    st.dataframe(preview, use_container_width=True, height=300)

try:
    data = apply_header(raw, int(header_row))
except Exception as exc:
    st.error(str(exc))
    st.stop()

if data.empty or len(data.columns) == 0:
    st.error("Header 적용 후 분석 가능한 데이터가 없습니다.")
    st.stop()

columns = list(data.columns)
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("분석 설정")
    x_col = st.selectbox("X 열 (독립변수)", columns, index=0)
    y_col = st.selectbox("Y 열 (종속변수)", columns, index=min(1, len(columns) - 1))
    error_col = st.selectbox("Y 오차 열 (선택)", [NONE] + columns, index=0)
    group_col = st.selectbox("사례 / 그룹 열 (선택)", [NONE] + columns, index=0)

    x_num = pd.to_numeric(data[x_col], errors="coerce")
    y_num = pd.to_numeric(data[y_col], errors="coerce")
    valid_pairs = int((x_num.notna() & y_num.notna()).sum())
    st.caption(f"유효한 숫자 X-Y 쌍: {valid_pairs:,} / {len(data):,}")

    st.markdown("**회귀모델 선택**")
    selected_models = st.multiselect(
        "분석할 모델",
        list(MODELS.keys()),
        default=["Linear", "Quadratic", "Cubic", "Exponential"]
    )
    show_ci = st.checkbox("95% 신뢰구간 표시", value=True)
    show_outliers = st.checkbox("이상치 후보 표시 (|표준화 잔차| ≥ 2)", value=True)

    run_clicked = st.button("▶ 회귀 분석 실행", type="primary", use_container_width=True)

with right:
    st.subheader("적용된 데이터")
    st.dataframe(data.head(100), use_container_width=True, height=420)

if run_clicked:
    if not selected_models:
        st.warning("하나 이상의 회귀모델을 선택하세요.")
        st.stop()
    results, failures = run_models(data, x_col, y_col, group_col, selected_models)
    st.session_state["analysis"] = {
        "results": results,
        "failures": failures,
        "data": data,
        "x_col": x_col,
        "y_col": y_col,
        "error_col": error_col,
        "group_col": group_col,
        "show_ci": show_ci,
        "show_outliers": show_outliers,
        "file_name": uploaded.name,
    }

analysis = st.session_state.get("analysis")
if not analysis:
    st.stop()

# Avoid showing stale results if the uploaded file changed.
if analysis.get("file_name") != uploaded.name:
    st.session_state.pop("analysis", None)
    st.info("새 파일이 선택되었습니다. 회귀 분석 실행 버튼을 눌러 다시 분석하세요.")
    st.stop()

results: list[FitResult] = analysis["results"]
failures: list[dict] = analysis["failures"]
if not results:
    st.error("모든 모델 분석에 실패했습니다.")
    if failures:
        st.dataframe(pd.DataFrame(failures), use_container_width=True)
    st.stop()

st.divider()
st.header("분석 결과")
summary = build_summary_df(results)
best = best_results(results)

# compact best-model cards
cases = list(best.keys())
card_cols = st.columns(min(max(len(cases), 1), 4))
for idx, case in enumerate(cases):
    r = best[case]
    with card_cols[idx % len(card_cols)]:
        st.metric(f"{case} · 최고 모델", MODEL_KR.get(r.model, r.model))
        st.caption(f"R² = {r.r2:.5f} · RMSE = {r.rmse:.5g}")

plot_tab, residual_tab, result_tab, outlier_tab, predict_tab, algorithm_tab = st.tabs(
    ["회귀 플롯", "잔차 플롯", "모델 결과", "이상치", "예측", "알고리즘"]
)

with plot_tab:
    fig = regression_figure(
        results, analysis["data"], analysis["x_col"], analysis["y_col"],
        analysis["error_col"], analysis["group_col"],
        analysis["show_ci"], analysis["show_outliers"]
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.markdown("**최고 모델 회귀식**")
    for case, r in best.items():
        st.code(f"{case} · {r.model}: {r.equation}", language=None)

with residual_tab:
    st.plotly_chart(residual_figure(results), use_container_width=True, config={"displaylogo": False})
    st.caption("Residual = Observed Y − Predicted Y. 이상치 후보는 |표준화 잔차| ≥ 2 기준입니다.")

with result_tab:
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "R²": st.column_config.NumberColumn(format="%.6f"),
            "Adjusted R²": st.column_config.NumberColumn(format="%.6f"),
            "RMSE": st.column_config.NumberColumn(format="%.6g"),
            "MAE": st.column_config.NumberColumn(format="%.6g"),
            "AICc": st.column_config.NumberColumn(format="%.4f"),
            "BIC": st.column_config.NumberColumn(format="%.4f"),
        }
    )
    with st.expander("파라미터 추정값 / 표준오차"):
        st.dataframe(build_parameter_df(results), use_container_width=True, hide_index=True)
    if failures:
        with st.expander(f"실패한 모델 {len(failures)}개"):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

with outlier_tab:
    best_resid = build_best_residual_df(results)
    outliers = best_resid[best_resid["Outlier"]].copy() if not best_resid.empty else best_resid
    if outliers.empty:
        st.success("현재 기준(|표준화 잔차| ≥ 2)에서 이상치 후보가 없습니다.")
    else:
        st.warning(f"이상치 후보 {len(outliers)}개가 검출되었습니다.")
        st.dataframe(outliers, use_container_width=True, hide_index=True)
    with st.expander("최고 모델의 전체 잔차 보기"):
        st.dataframe(best_resid, use_container_width=True, hide_index=True)

with predict_tab:
    prediction_x = st.number_input("예측할 X 값", value=0.0, format="%.8g")
    if st.button("예측 계산"):
        pred_rows = []
        for case, r in best.items():
            if r.spec.positive_x and prediction_x <= 0:
                pred_rows.append({"Case": case, "Model": r.model, "Predicted Y": "X must be > 0"})
            else:
                value = float(r.predict(np.array([prediction_x], dtype=float))[0])
                pred_rows.append({"Case": case, "Model": r.model, "Predicted Y": value})
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

with algorithm_tab:
    st.info("이 탭은 과제의 '프로그램 알고리즘 설명'을 위해 실제 코드에 사용된 계산 절차를 정리한 것입니다.")
    render_algorithm_overview(sorted(set(r.model for r in results), key=list(MODELS.keys()).index))

st.subheader("결과 다운로드")
excel_bytes = export_excel_bytes(results, failures, analysis["data"])
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "⬇ 분석 결과 Excel 다운로드",
        data=excel_bytes,
        file_name="regression_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with col2:
    html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        "⬇ 회귀 그래프 HTML 다운로드",
        data=html_bytes,
        file_name="regression_plot.html",
        mime="text/html",
        use_container_width=True,
    )

st.caption(
    "알고리즘: 선형/2차/3차 다항 회귀는 최소제곱 적합, "
    "지수/로그/거듭제곱/4-모수 로지스틱 회귀는 SciPy curve_fit의 비선형 최소제곱(TRF) 최적화를 사용합니다. "
    "최고 모델은 AICc(계산 불가 시 AIC)가 가장 낮은 모델 기준입니다."
)
