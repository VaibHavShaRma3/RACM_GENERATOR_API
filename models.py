from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobPhase(str, Enum):
    queued = "queued"
    extracting = "extracting"
    chunking = "chunking"
    analyzing = "analyzing"
    consolidating = "consolidating"
    deduplicating = "deduplicating"
    completed = "completed"
    failed = "failed"


RACM_FIELDS = [
    "Process Area",
    "Sub-Process",
    "Risk ID",
    "Risk Description",
    "Risk Category",
    "Risk Type",
    "Control ID",
    "Control Activity",
    "Control Objective",
    "Control Type",
    "Control Nature",
    "Control Frequency",
    "Control Owner",
    "Control description as per SOP",
    "Testing Attributes",
    "Evidence/Source",
    "Assertion Mapped",
    "Compliance Reference",
    "Risk Likelihood",
    "Risk Impact",
    "Risk Rating",
    "Mitigation Effectiveness",
    "Gaps/Weaknesses Identified",
]


class RACMEntry(BaseModel):
    process_area: str = Field(default="", alias="Process Area")
    sub_process: str = Field(default="", alias="Sub-Process")
    risk_id: str = Field(default="", alias="Risk ID")
    risk_description: str = Field(default="", alias="Risk Description")
    risk_category: str = Field(default="", alias="Risk Category")
    risk_type: str = Field(default="", alias="Risk Type")
    control_id: str = Field(default="", alias="Control ID")
    control_activity: str = Field(default="", alias="Control Activity")
    control_objective: str = Field(default="", alias="Control Objective")
    control_type: str = Field(default="", alias="Control Type")
    control_nature: str = Field(default="", alias="Control Nature")
    control_frequency: str = Field(default="", alias="Control Frequency")
    control_owner: str = Field(default="", alias="Control Owner")
    control_description_sop: str = Field(default="", alias="Control description as per SOP")
    testing_attributes: str = Field(default="", alias="Testing Attributes")
    evidence_source: str = Field(default="", alias="Evidence/Source")
    assertion_mapped: str = Field(default="", alias="Assertion Mapped")
    compliance_reference: str = Field(default="", alias="Compliance Reference")
    risk_likelihood: str = Field(default="", alias="Risk Likelihood")
    risk_impact: str = Field(default="", alias="Risk Impact")
    risk_rating: str = Field(default="", alias="Risk Rating")
    mitigation_effectiveness: str = Field(default="", alias="Mitigation Effectiveness")
    gaps_weaknesses: str = Field(default="", alias="Gaps/Weaknesses Identified")

    model_config = {"populate_by_name": True}


class RACMResponse(BaseModel):
    detailed_entries: list[RACMEntry]
    summary_entries: list[RACMEntry]


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    phase: JobPhase
    progress_pct: int
    progress_msg: str
    created_at: str
    updated_at: str
    file_name: str


class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[RACMResponse] = None
    error: Optional[str] = None
