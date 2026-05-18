import asyncio
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app import config
from app.hazards import drought, precipitation, sea_level, temperature

app = FastAPI(
    title="Climate Hazard Assessment API",
    description="Baseline-delta hazard risk scoring. Baseline: 2000-2020 | Future: 2020-2040",
    version="0.1.0",
)

Scenario = Literal["ssp126", "ssp245", "ssp585"]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class RiskDetail(BaseModel):
    score: int = Field(..., ge=0, le=100)
    level: str


class TemperatureResult(BaseModel):
    hazard: str = "temperature"
    baseline_period: str = "2000-2020"
    future_period: str = "2020-2040"
    baseline_mean_c: float
    future_mean_c: float
    delta_c: float
    risk: RiskDetail


class PrecipitationResult(BaseModel):
    hazard: str = "precipitation"
    baseline_period: str = "2000-2020"
    future_period: str = "2020-2040"
    baseline_p99_mm_day: float
    future_p99_mm_day: float
    delta_pct: float
    risk: RiskDetail


class DroughtResult(BaseModel):
    hazard: str = "drought"
    baseline_period: str = "2000-2020"
    future_period: str = "2020-2040"
    baseline_dry_wetness: float
    future_dry_wetness: float
    delta_wetness: float
    risk: RiskDetail


class SeaLevelResult(BaseModel):
    hazard: str = "sea_level"
    baseline_period: str = "2000-2020"
    future_period: str = "2020-2040"
    baseline_rate_mm_yr: float
    projected_delta_cm: float
    risk: RiskDetail


class FullAssessment(BaseModel):
    lat: float
    lon: float
    scenario: str
    temperature: TemperatureResult
    precipitation: PrecipitationResult
    drought: DroughtResult
    sea_level: SeaLevelResult
    composite_risk_score: int = Field(..., description="Unweighted mean of all hazard scores")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/assessment/temperature", response_model=TemperatureResult)
async def assess_temperature(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    scenario: Scenario = config.DEFAULT_SCENARIO,
):
    try:
        result = await asyncio.to_thread(temperature.assess, lat, lon, scenario)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return TemperatureResult(
        baseline_mean_c=result.baseline_mean_c,
        future_mean_c=result.future_mean_c,
        delta_c=result.delta_c,
        risk=RiskDetail(score=result.risk.score, level=result.risk.level),
    )


@app.get("/assessment/precipitation", response_model=PrecipitationResult)
async def assess_precipitation(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    scenario: Scenario = config.DEFAULT_SCENARIO,
):
    try:
        result = await asyncio.to_thread(precipitation.assess, lat, lon, scenario)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return PrecipitationResult(
        baseline_p99_mm_day=result.baseline_p99_mm_day,
        future_p99_mm_day=result.future_p99_mm_day,
        delta_pct=result.delta_pct,
        risk=RiskDetail(score=result.risk.score, level=result.risk.level),
    )


@app.get("/assessment/drought", response_model=DroughtResult)
async def assess_drought(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    scenario: Scenario = config.DEFAULT_SCENARIO,
):
    try:
        result = await asyncio.to_thread(drought.assess, lat, lon, scenario)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return DroughtResult(
        baseline_dry_wetness=result.baseline_dry_wetness,
        future_dry_wetness=result.future_dry_wetness,
        delta_wetness=result.delta_wetness,
        risk=RiskDetail(score=result.risk.score, level=result.risk.level),
    )


@app.get("/assessment/sea_level", response_model=SeaLevelResult)
async def assess_sea_level(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    scenario: Scenario = config.DEFAULT_SCENARIO,
):
    try:
        result = await asyncio.to_thread(sea_level.assess, lat, lon, scenario)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return SeaLevelResult(
        baseline_rate_mm_yr=result.baseline_rate_mm_yr,
        projected_delta_cm=result.projected_delta_cm,
        risk=RiskDetail(score=result.risk.score, level=result.risk.level),
    )


@app.get("/assessment", response_model=FullAssessment)
async def full_assessment(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    scenario: Scenario = config.DEFAULT_SCENARIO,
):
    """Run all four hazard assessments concurrently and return composite score."""
    try:
        t, p, d, s = await asyncio.gather(
            asyncio.to_thread(temperature.assess, lat, lon, scenario),
            asyncio.to_thread(precipitation.assess, lat, lon, scenario),
            asyncio.to_thread(drought.assess, lat, lon, scenario),
            asyncio.to_thread(sea_level.assess, lat, lon, scenario),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    scores = [t.risk.score, p.risk.score, d.risk.score, s.risk.score]
    composite = round(sum(scores) / len(scores))

    return FullAssessment(
        lat=lat,
        lon=lon,
        scenario=scenario,
        temperature=TemperatureResult(
            baseline_mean_c=t.baseline_mean_c,
            future_mean_c=t.future_mean_c,
            delta_c=t.delta_c,
            risk=RiskDetail(score=t.risk.score, level=t.risk.level),
        ),
        precipitation=PrecipitationResult(
            baseline_p99_mm_day=p.baseline_p99_mm_day,
            future_p99_mm_day=p.future_p99_mm_day,
            delta_pct=p.delta_pct,
            risk=RiskDetail(score=p.risk.score, level=p.risk.level),
        ),
        drought=DroughtResult(
            baseline_dry_wetness=d.baseline_dry_wetness,
            future_dry_wetness=d.future_dry_wetness,
            delta_wetness=d.delta_wetness,
            risk=RiskDetail(score=d.risk.score, level=d.risk.level),
        ),
        sea_level=SeaLevelResult(
            baseline_rate_mm_yr=s.baseline_rate_mm_yr,
            projected_delta_cm=s.projected_delta_cm,
            risk=RiskDetail(score=s.risk.score, level=s.risk.level),
        ),
        composite_risk_score=composite,
    )
